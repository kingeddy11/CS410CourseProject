import pickle
from nltk.corpus import stopwords
import pandas as pd
from collections import defaultdict, Counter
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import math
import numpy as np


print("Loading data...")
df = pd.read_csv('../../data/yelp_reviews_users_Phila_final.csv')
stopwords_english = set(stopwords.words('english'))

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

# Function to lemmatize words and count words in a text
def count_words(text):
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

# Initialize tf dictionary
tf = defaultdict(lambda: defaultdict(int))

# Group by 'business_id' and iterate through the groups
count = 0
for business_id, group in df.groupby('business_id'):
    word_counter = Counter()  # Accumulate word counts for the current business_id
    for text in group['review']:
        word_counter.update(count_words(text))  # Update word count
    
    tf[business_id] = dict(word_counter)  # Convert the counter to a dict and store it
    print(f"Counting tf for doc {count} (ID:{business_id})")
    count += 1

tf = dict(tf)

# Construct idf dictionary by applying log_2((M + 1) / df(w))
print("Counting doc frequency...")

# Counting total number of documents
M = len(df['business_id'])

# Counting number of docs that contain word w
df_w = dict()

for _, term_dict in tf.items():
    for term, _ in term_dict.items():
        if term in df_w:
            df_w[term] += 1
        else:
            df_w[term] = 1

# Calculating the idf
idf = {term: math.log2((M + 1 / df_w[term])) for term in df_w}

# Construct document length normalizer
doc_length_normalizer = dict() # Initialize document length normalizer dictionary

# Set document length normalizer parameter (penalization for long documents)
b = 0.5

# Calculate average document length by taking the average of the sum of all word counts in tf dictionary for each business_id
word_counts = np.array([sum(term_dict.values()) for term_dict in tf.values()])
avg_doc_length = np.mean(word_counts)

# Calculate document length normalizer for each business_id
for business_id, term_dict in tf.items():
    doc_length = sum(term_dict.values()) # Calculate document length for each business_id
    doc_length_normalizer[business_id] = 1 / (1 - b + b * (doc_length / avg_doc_length))



print("Saving files...")
with open("yelp_reviews_tfidf_dl_norm_piv_len_norm.pkl", 'wb') as file:
    pickle.dump((tf, idf, doc_length_normalizer), file)