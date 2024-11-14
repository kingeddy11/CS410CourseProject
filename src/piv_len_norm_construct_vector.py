import pickle
import numpy as np
from gensim.models import KeyedVectors

print("Loading data...")
with open("yelp_reviews_tfidf_dl_norm_piv_len_norm.pkl", 'rb') as file:
    tf, idf = pickle.load(file)

## Construct document length normalizer
print("Computing document length normalizer...")
doc_length_normalizer = dict() # Initialize document length normalizer dictionary

# Set document length normalizer parameter (penalization for long documents)
b = 0.9 # Setting larger normalizer to address the issue of businesses with more reviews and longer reviews (long documents)

# Calculate average document length by taking the average of the sum of all word counts in tf dictionary for each business_id
word_counts = np.array([sum(term_dict.values()) for term_dict in tf.values()])
avg_doc_length = np.mean(word_counts)

# Calculate document length normalizer for each business_id
for business_id, term_dict in tf.items():
    doc_length = sum(term_dict.values()) # Calculate document length for each business_id
    doc_length_normalizer[business_id] = (1 - b + b * (doc_length / avg_doc_length))


## Construct document vectors
print("Constructing doc vector...")
doc_vectors = dict()
for business_id, term_freqs in tf.items():
    doc_vector = []
    for term, freq in term_freqs.items():
        tf_weight = np.log(1 + np.log(1 + freq))
        weight = (tf_weight / doc_length_normalizer[business_id]) * idf[term]
        doc_vector.append((term, weight))

    doc_vectors[business_id] = doc_vector

# doc_vectors = dict()
# for business_id, term_freqs in tf.items():
#     vectors = []
#     total = 0
#     for term, freq in term_freqs.items():
#         tf_weight = np.log(1 + np.log(1 + freq))
#         weight = (tf_weight / doc_length_normalizer[business_id]) * idf[term]
#         vectors.append((weight, term))
#         total += weight
            
#     if total > 0:
#         doc_vector = {term: weight / total for weight, term in vectors} # Normalize the weights by total weight for each business_id to address the issue of businesses with more reviews and longer reviews
#         doc_vectors[business_id] = doc_vector

print("Saving files...")
with open("yelp_reviews_doc_vectors_piv_len_norm.pkl", 'wb') as file:
    pickle.dump(doc_vectors, file)