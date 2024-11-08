from gensim.models import KeyedVectors
import numpy as np
from nltk.corpus import stopwords
import re
from collections import Counter
import pickle

print("Loading data...")
stopwords_english = set(stopwords.words('english'))
model_path = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

with open("yelp_reviews_doc_vectors.pkl", 'rb') as file:
    vector_dict = pickle.load(file)

# Function to compute the weighted average word2vec vector
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

def count_words(text):
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words and convert to lowercase
    return Counter(word for word in words if word not in stopwords_english)

QUERY = "Turkey mixed grill"
print("Constructing query vector...")
queryv = construct_query_vector(count_words(QUERY))

print("Searching...")
scores = []
for bid, vector in vector_dict.items():
    scores.append((bid, np.dot(queryv, vector)))

# Sort vectors by dot product value in descending order
ranked_vectors = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
print(ranked_vectors)