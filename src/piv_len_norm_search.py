import numpy as np
from nltk.corpus import stopwords
import re
from collections import Counter
import pickle
import csv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Loading data...")
stopwords_english = set(stopwords.words('english'))

with open("yelp_reviews_doc_vectors_piv_len_norm.pkl", 'rb') as file:
    vector_dict = pickle.load(file)

print("Loading sentiment analysis data...")
lexicon_sentiment_df = pd.read_csv('../../data/yelp_restaurants_lexicon_sentiment_Phila.csv')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to get WordNet POS tag
def getwordnet_pos(pos):
    if pos.startswith('N'):
        return wn.NOUN
    elif pos.startswith('V'):
        return wn.VERB
    elif pos.startswith('J'):
        return wn.ADJ
    elif pos.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN # Default to noun

# Function to lemmatize words and count words in query and construct query dict
def construct_query_vector(text):
    # Extract words and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())

    # POS tag words
    words_pos = pos_tag(words)

    # Lemmatizing words
    lemmatized_words = []
    for word, pos in words_pos:
        wordnet_pos = getwordnet_pos(pos)
        lemmatized_words.append(lemmatizer.lemmatize(word, pos = wordnet_pos))
    
    return Counter(word for word in lemmatized_words if word not in stopwords_english)


# Input query
print("Please enter a query: ")
QUERY = input()
QUERY = QUERY.lower() # convert query to lowercase
print("Constructing query vector...")
query_vector = construct_query_vector(QUERY)


# Function to calculate the dot product to compute similarity between two vectors
def calc_dot_product(query_vector, doc_vector):
    dot_product = 0
    for term, weight in doc_vector:
        if term in query_vector:
            dot_product += query_vector[term] * weight
    
    return dot_product


# Implement business_id (restaurant) retrieval by calculating similarity scores between query and document vector for each business id
# higher similarity score indicates higher relevance between query and document vector
print("Searching...")
sim_scores = []
for bid, doc_vector in vector_dict.items():
    sim_scores.append((bid, calc_dot_product(query_vector, doc_vector)))

sim_scores_df = pd.DataFrame(sim_scores, columns = ['business_id', 'sim_score']) # convert similarity scores to a pandas dataframe


# Normalized similarity score, vader sentiment scores, TextBlob sentiment scores, sentiwordnet sentiment scores by calculating z-scores for each variable;
# addresses the issue of different scales and distributions of similarity scores and sentiment scores 
z_score_scaler_sim = StandardScaler()
sim_scores_df['norm_sim_score'] = z_score_scaler_sim.fit_transform(sim_scores_df['sim_score'].values.reshape(-1, 1))

# Normalized averaged sentiment scores by calculating z-scores for each variable, averaging the 3 normalized sentiment scores, and then calculating z-scores for the averaged sentiment score;
# addresses the issue of different scales and distributions of similarity scores and sentiment scores for each of the models and lowers the possibility of one model dominating the weighted score
z_score_scaler_avg_sentiment = StandardScaler()
lexicon_sentiment_df[['norm_avg_sentiment_vader', 'norm_avg_sentiment_TextBlob', 'norm_avg_sentiment_sentiwordnet']] = z_score_scaler_avg_sentiment.fit_transform(lexicon_sentiment_df[['avg_sentiment_vader', 'avg_sentiment_TextBlob', 'avg_sentiment_sentiwordnet']])
lexicon_sentiment_df['norm_avg_sentiment'] = lexicon_sentiment_df[['norm_avg_sentiment_vader', 'norm_avg_sentiment_TextBlob', 'norm_avg_sentiment_sentiwordnet']].mean(axis = 1)
lexicon_sentiment_df['norm_avg_sentiment'] = z_score_scaler_avg_sentiment.fit_transform(lexicon_sentiment_df['norm_avg_sentiment'].values.reshape(-1, 1))

# Normalized weighted averaged sentiment scores by calculating z-scores for each variable, averaging the 3 normalized sentiment scores, and then calculating z-scores for the averaged sentiment score;
# addresses the issue of different scales and distributions of similarity scores and sentiment scores for each of the models and lowers the possibility of one model dominating the weighted score
z_score_scaler_wght_avg_sentiment = StandardScaler()
lexicon_sentiment_df[['norm_wght_avg_sentiment_vader', 'norm_wght_avg_sentiment_TextBlob', 'norm_wght_avg_sentiment_sentiwordnet']] = z_score_scaler_wght_avg_sentiment.fit_transform(lexicon_sentiment_df[['weighted_avg_sentiment_vader', \
                                                                                                                                                                                                            'weighted_avg_sentiment_TextBlob', \
                                                                                                                                                                                                            'weighted_avg_sentiment_sentiwordnet']])
lexicon_sentiment_df['norm_wght_avg_sentiment'] = lexicon_sentiment_df[['norm_wght_avg_sentiment_vader', 'norm_wght_avg_sentiment_TextBlob', 'norm_wght_avg_sentiment_sentiwordnet']].mean(axis = 1)
lexicon_sentiment_df['norm_wght_avg_sentiment'] = z_score_scaler_wght_avg_sentiment.fit_transform(lexicon_sentiment_df['norm_wght_avg_sentiment'].values.reshape(-1, 1))


# Merging scores with sentiment analysis data
scores_df = pd.merge(sim_scores_df, lexicon_sentiment_df, on= 'business_id', how = 'inner')
print('Count of unique business ids', len(scores_df['business_id'].unique())) # Checking to see if all 200 business ids are present after merging

"""=======Hyperparameters to vary======="""
# Set the weights for how much influence each variable has on the weighted score; parameters should sum to 1 and be between 0 and 1
sim_score_wght = 0.8
sentiment_wght = 0.2

# Specify whether you want to calculate the weighted score using weighted average sentiment scores or average sentiment scores
weighted_avg_sentiment = True
"""====================================="""

# Calculating a weighted average score of similarity and average sentiment scores for each business id
def calc_weighted_score(row, sim_score_wght = sim_score_wght, sentiment_wght = sentiment_wght, weighted_avg_sentiment = weighted_avg_sentiment):
    # weighted average sentiment scores calculation
    if weighted_avg_sentiment:
        return (sim_score_wght * row['norm_sim_score'] + 
                sentiment_wght * row['norm_wght_avg_sentiment'])
    
    # average sentiment scores calculation
    else:
        return (sim_score_wght * row['norm_sim_score'] + 
                sentiment_wght * row['norm_avg_sentiment'])
    

scores_df['weighted_score'] = scores_df.apply(calc_weighted_score, axis = 1)

# Sort vectors by weighted score in descending order and grab relevant columns depending on whether you are using weighted average sentiment scores or average sentiment scores
if weighted_avg_sentiment:
    ranked_weight_scores = scores_df.sort_values(by = 'weighted_score', ascending = False)[['business_id', 'weighted_score', 'sim_score', 'norm_sim_score', 'norm_wght_avg_sentiment', 'norm_wght_avg_sentiment_vader', 'norm_wght_avg_sentiment_TextBlob', 'norm_wght_avg_sentiment_sentiwordnet']]
else:
    ranked_weight_scores = scores_df.sort_values(by = 'weighted_score', ascending = False)[['business_id', 'weighted_score', 'sim_score', 'norm_sim_score', 'norm_avg_sentiment', 'norm_avg_sentiment_vader', 'norm_avg_sentiment_TextBlob', 'norm_avg_sentiment_sentiwordnet']]

# Top 10 business ids with highest weighted score
ranked_weight_scores_top10 = ranked_weight_scores[:10]
print(ranked_weight_scores_top10)

# Write the results of either weighted average sentiment scores or average sentiment scores to a CSV file
if weighted_avg_sentiment:
    ranked_weight_scores.to_csv(f'../../data/restaurant_ranking_pln_w_avg_sentiment_{QUERY}.csv', index = False)
else:
    ranked_weight_scores.to_csv(f'../../data/restaurant_ranking_pln_avg_sentiment_{QUERY}.csv', index = False)
