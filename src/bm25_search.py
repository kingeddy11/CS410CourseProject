import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np

print("Loading data...")
with open("yelp_reviews_bm25.pkl", 'rb') as file:
    bm25, business_index = pickle.load(file)
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

print("Searching...")
QUERY = "japanese"
tokenized_query = [ps.stem(w) for w in tokenizer.tokenize(QUERY)]
scores = bm25.get_scores(tokenized_query)
descending_indices = np.argsort(scores)[::-1]
result = [business_index[i] for i in descending_indices]
print(result[:10])