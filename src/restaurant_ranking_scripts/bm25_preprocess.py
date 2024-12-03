import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import pickle
# import nltk
# nltk.download('punkt')

print("Loading data...")
df = pd.read_csv('../../data/data_cleaning/yelp_reviews_users_Phila_final.csv')
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

print("Tokenize text...")
corpus = []
business_index = []
for business_id, group in df.groupby('business_id'):
    print(f'Tokenizing reviews for business_id: {business_id}')
    result = []
    for text in group['review']:
        result += [ps.stem(w) for w in tokenizer.tokenize(text)]
    corpus.append(result)
    business_index.append(business_id)

print("Saving file...")
with open("yelp_reviews_preprocess_bm25.pkl", 'wb') as file:
    pickle.dump((corpus, business_index), file)