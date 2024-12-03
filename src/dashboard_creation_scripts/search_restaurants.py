import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter
from gensim.models import KeyedVectors

model_path = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def getwordnet_pos(pos):
    """Map POS tag to WordNet POS for lemmatization."""
    if pos.startswith('N'):
        return wn.NOUN
    elif pos.startswith('V'):
        return wn.VERB
    elif pos.startswith('J'):
        return wn.ADJ
    elif pos.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN  # Default to noun

def preprocess_query(query):
    """Tokenize, stem/lemmatize, and remove stopwords."""
    stopwords_english = set(stopwords.words('english'))
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenize and lemmatize/stem query
    tokens = tokenizer.tokenize(query.lower())
    pos_tags = pos_tag(tokens)
    processed_tokens = [
        lemmatizer.lemmatize(word, getwordnet_pos(pos))
        for word, pos in pos_tags if word not in stopwords_english
    ]
    return processed_tokens
    
# Calculates the doc vector with pivoted length normalization
def calculate_doc_length_normalization(doc_vector, doc_length_normalized, B):
    normalized_vector = []
    normalization_factor = (1 - B + B * doc_length_normalized)  # Avoid recalculating
    
    for term, weight in doc_vector:
        if normalization_factor != 0:  # Ensure no division by zero
            normalized_weight = weight / normalization_factor
        else:
            normalized_weight = 0  # Safeguard against zero normalization factors
        normalized_vector.append((term, normalized_weight))
    
    return normalized_vector

# Function (for pln) to calculate the dot product to compute similarity between query vector and document vector for each business id
def calc_dot_product(query_vector, length_normalized_doc_vector):
    dot_product = 0
    for term, weight in length_normalized_doc_vector:
        if term in query_vector:
            dot_product += query_vector[term] * weight
    
    return dot_product

# Functions for word2vec preprocessing
def count_words(text):
    # Extract words and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())

    # POS tag words
    words_pos = pos_tag(words)

    # Lemmatizing words
    stopwords_english = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word, pos in words_pos:
        wordnet_pos = getwordnet_pos(pos)
        lemmatized_words.append(lemmatizer.lemmatize(word, pos = wordnet_pos))

    return Counter(word for word in lemmatized_words if word not in stopwords_english)

def construct_query_vector(word_count_dict):
    # Initialize an empty vector of the same size as Word2Vec embeddings
    weighted_sum = np.zeros(model.vector_size)
    total_count = 0
    
    # For each word and its count in the dictionary
    for word, count in word_count_dict.items():
        if word in model:
            # Get the word's vector and multiply it by the count (weight)
            word_vector = model[word]
            weighted_sum += word_vector * count
            total_count += count
    
    # Compute the weighted average vector (avoid division by zero)
    if total_count > 0:
        weighted_avg_vector = weighted_sum / total_count
    else:
        weighted_avg_vector = np.zeros(model.vector_size)
    
    return weighted_avg_vector

# Retrieval Functions
def bm25_scoring(tokens, bm25, business_index):
    print("Using BM25 scoring...")
    scores = bm25.get_scores(tokens)
    descending_indices = np.argsort(scores)[::-1]
    return pd.DataFrame([(business_index[i], scores[i]) for i in descending_indices], columns=['business_id', 'sim_score'])

def pln_scoring(tokens, vector_dict, doc_length_normalizer, B = 0.9):
    print("Using Pivoted Length Normalization scoring...")
    sim_scores = []
    query_vector = Counter(tokens)
    for bid, doc_vector in vector_dict.items():
        length_normalized_doc_vector = calculate_doc_length_normalization(doc_vector, doc_length_normalizer[bid], B)
        sim_scores.append((bid, calc_dot_product(query_vector, length_normalized_doc_vector)))
    
    return pd.DataFrame(sim_scores, columns = ['business_id', 'sim_score'])

def w2c_scoring(query_vector, vector_dict):
    sim_scores = []
    for bid, vector in vector_dict.items():
        sim_scores.append((bid, np.dot(query_vector, vector)))
        
    return pd.DataFrame(sim_scores, columns = ['business_id', 'sim_score'])
    
# Main Search Function
def search_restaurants(query, method="bm25", sim_score_wght=0.8, weighted_avg_sentiment=True, num_results = 10, B = 0.9):
    print("Loading data...")
    with open("../restaurant_ranking_scripts/yelp_reviews_preprocess_bm25.pkl", 'rb') as bm25_file:
        bm25, business_index = pickle.load(bm25_file)
              
    with open("../restaurant_ranking_scripts/yelp_reviews_doc_vectors_pln.pkl", 'rb') as pln_file:
        vector_dict, doc_length_normalizer = pickle.load(pln_file)

    with open("../restaurant_ranking_scripts/yelp_reviews_doc_vectors_word2vec.pkl", 'rb') as file:
        vector_dict_w2v = pickle.load(file)

    lexicon_sentiment_df = pd.read_csv('../../data/sentiment_analysis/yelp_restaurants_lexicon_sentiment_Phila.csv')
    bert_sentiment_df = pd.read_csv('../../data/sentiment_analysis/yelp_restaurants_bert_sentiment_Phila.csv')
    bert_sentiment_df.columns = ['business_id', 'avg_sentiment_bert', 'weighted_avg_sentiment_bert', 
                                 'negative_review_count_bert', 'neutral_review_count_bert', 'positive_review_count_bert']
    sentiment_df = pd.merge(lexicon_sentiment_df, bert_sentiment_df, on='business_id', how='inner')

    user_reviews_df = pd.read_csv('../../data/data_cleaning/yelp_restaurants_Phila_final.csv')

    # Preprocess Query (for pln and bm25)
    query_tokens = preprocess_query(query)

    # Preprocess Query (for word2vec)
    queryv = construct_query_vector(count_words(query))
    
    # Select Retrieval Method
    if method == "bm25":
        sim_scores_df = bm25_scoring(query_tokens, bm25, business_index)
    elif method == "pln":
        sim_scores_df = pln_scoring(query_tokens, vector_dict, doc_length_normalizer, B)
    elif method == "word2vec":
        sim_scores_df = w2c_scoring(queryv, vector_dict_w2v)
    else:
        raise ValueError("Invalid method. Choose 'bm25' or 'pln'.")
    
    # Normalize Scores
    z_score_scaler_sim = StandardScaler()
    sim_scores_df['norm_sim_score'] = z_score_scaler_sim.fit_transform(sim_scores_df['sim_score'].values.reshape(-1, 1))

    # Normalize Sentiment Scores
    sentiment_cols = ['avg_sentiment_vader', 'avg_sentiment_TextBlob', 'avg_sentiment_sentiwordnet', 'avg_sentiment_bert']
    sentiment_cols_wght = ['norm_wght_avg_sentiment_vader', 'norm_wght_avg_sentiment_TextBlob', 'norm_wght_avg_sentiment_sentiwordnet', 'norm_wght_avg_sentiment_bert']    
    z_score_scaler_sentiment = StandardScaler()
    sentiment_df[[f'norm_{col}' for col in sentiment_cols]] = z_score_scaler_sentiment.fit_transform(sentiment_df[sentiment_cols])
    sentiment_df['norm_avg_sentiment'] = sentiment_df[[f'norm_{col}' for col in sentiment_cols]].mean(axis=1)
    
    sentiment_df[[f'norm_{col}' for col in sentiment_cols_wght]] = z_score_scaler_sentiment.fit_transform(sentiment_df[['weighted_avg_sentiment_vader', \
                                                                                                                                                                                                            'weighted_avg_sentiment_TextBlob', \
                                                                                                                                                                                                            'weighted_avg_sentiment_sentiwordnet', \
                                                                                                                                                                                                            'weighted_avg_sentiment_bert']])
    sentiment_df['norm_wght_avg_sentiment'] = sentiment_df[[f'norm_{col}' for col in sentiment_cols_wght]].mean(axis = 1)
    sentiment_df['norm_wght_avg_sentiment'] = z_score_scaler_sentiment.fit_transform(sentiment_df['norm_wght_avg_sentiment'].values.reshape(-1, 1))

    # Merge Scores with Sentiment Data
    scores_df = pd.merge(sim_scores_df, sentiment_df, on='business_id', how='inner')

    # Calculate Weighted Score    
    sentiment_weight = 1 - sim_score_wght
    # weighted average sentiment scores calculation
    if weighted_avg_sentiment:
        scores_df['weighted_score'] =  (sim_score_wght * scores_df['norm_sim_score'] + 
                sentiment_weight * scores_df['norm_wght_avg_sentiment'])
    
    # average sentiment scores calculation
    else:
        scores_df['weighted_score'] =  (sim_score_wght * scores_df['norm_sim_score'] + 
                sentiment_weight * scores_df['norm_avg_sentiment'])
            
    # Rank and Return Top Results
    ranked_scores = scores_df.sort_values(by='weighted_score', ascending=False)
    return ranked_scores.merge(user_reviews_df[['business_id', 'restaurant_name']], on='business_id', how='inner')[['business_id', 'restaurant_name', 'weighted_score']].head(num_results)


# CLI Execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        query = sys.argv[1]
        method = sys.argv[2]
        print(f"Processing query: {query} using {method} method.")
        top_10 = search_restaurants(query, method)
        print("\nTop 10 Restaurants:")
        print(top_10.to_string(index=False))
    else:
        print("Usage: python search_restaurants.py '<query>' <method>")
        print("Method must be 'bm25' or 'pln'.")
