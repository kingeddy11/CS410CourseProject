import pickle
import numpy as np
from gensim.models import KeyedVectors

print("Loading data...")
model_path = '../../models/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

with open("yelp_reviews_preprocess_word2vec.pkl", 'rb') as file:
    tf, idf = pickle.load(file)

print("Constructing doc vector...")
doc_vectors = dict()
for business_id, term_freqs in tf.items():
    vectors = []
    total = 0
    for term, freq in term_freqs.items():
        if term in model:
            weight = (freq + 1) / idf[term]
            vectors.append((weight, term))
            total += weight

    doc_vector = np.zeros(model.vector_size)
    for weight, word in vectors:
        word_vector = model[word]
        doc_vector += word_vector * (weight/total)
    doc_vectors[business_id] = doc_vector

print("Saving files...")
with open("yelp_reviews_doc_vectors_word2vec.pkl", 'wb') as file:
    pickle.dump(doc_vectors, file)
