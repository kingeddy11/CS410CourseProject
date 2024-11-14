from gensim.models import KeyedVectors
import numpy as np
from nltk.corpus import stopwords
import re
from collections import Counter
import pickle
import csv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag

print("Loading data...")
stopwords_english = set(stopwords.words('english'))

with open("yelp_reviews_doc_vectors_piv_len_norm.pkl", 'rb') as file:
    vector_dict = pickle.load(file)

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
print("Constructing query vector...")
query_vector = construct_query_vector(QUERY)


# Function to calculate the dot product to compute similarity between two vectors
def calc_dot_product(query_vector, doc_vector):
    dot_product = 0
    for term, weight in doc_vector:
        if term in query_vector:
            dot_product += query_vector[term] * weight
    
    return dot_product


# Implement business_id (restaurant) retrieval
print("Searching...")
scores = []
for bid, doc_vector in vector_dict.items():
    scores.append((bid, calc_dot_product(query_vector, doc_vector)))

# Sort vectors by dot product value in descending order
ranked_vectors = sorted(scores, key=lambda x: x[1], reverse=True)
ranked_vectors_top10 = ranked_vectors[:10]
print(ranked_vectors_top10)

# Write the results to a CSV file
with open(f'../../data/yelp_reviews_search_results_piv_len_norm_{QUERY}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["business_id", "score"])
    for bid, score in ranked_vectors:
        writer.writerow([bid, score])