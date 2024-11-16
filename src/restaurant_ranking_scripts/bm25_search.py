import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

print("Loading data...")
with open("yelp_reviews_preprocess_bm25.pkl", 'rb') as file:
    bm25, business_index = pickle.load(file)

print("Loading sentiment analysis data...")
lexicon_sentiment_df = pd.read_csv('../../data/sentiment_analysis/yelp_restaurants_lexicon_sentiment_Phila.csv')

# Initalizing stemmer and tokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Input query
print("Please enter a query: ")
QUERY = input()
QUERY = QUERY.lower() # convert query to lowercase

# Tokenize and stem the query
print("Searching...")
tokenized_query = [ps.stem(w) for w in tokenizer.tokenize(QUERY)]

# Implement business_id (restaurant) retrieval by computing similarity scores using BM25
scores = bm25.get_scores(tokenized_query)
descending_indices = np.argsort(scores)[::-1]
sim_scores = [(business_index[i], scores[i]) for i in descending_indices]
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


# Calculating a weighted average score of similarity and (weighted) average sentiment scores for each business id
"""=======Hyperparameters to vary======="""
# Set the weight for similarity scores to indicate how much influence it has on the weighted score; the weight should be in the range of 0 to 1
sim_score_wght = 0.8

# Specify whether you want to calculate the weighted score using weighted average sentiment scores or average sentiment scores
weighted_avg_sentiment = True
"""====================================="""

def calc_weighted_score(row, sim_score_wght = sim_score_wght, weighted_avg_sentiment = weighted_avg_sentiment):
    # computing the sentiment score weight as the complement of the similarity score weight
    sentiment_wght = 1 - sim_score_wght

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
    ranked_weight_scores.to_csv(f'../../data/restaurant_rankings_by_query/restaurant_ranking_bm25_w_avg_sentiment_{QUERY}.csv', index = False)
else:
    ranked_weight_scores.to_csv(f'../../data/restaurant_rankings_by_query/restaurant_ranking_bm25_avg_sentiment_{QUERY}.csv', index = False)