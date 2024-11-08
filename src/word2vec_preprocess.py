import pickle
from nltk.corpus import stopwords
import pandas as pd
from collections import defaultdict, Counter
import re

print("Loading data...")
df = pd.read_csv('yelp_reviews_users_Phila_final.csv')
stopwords_english = set(stopwords.words('english'))

# Function to count words in a text
def count_words(text):
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words and convert to lowercase
    return Counter(word for word in words if word not in stopwords_english)

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

# Construct idf dictionary
print("Counting doc frequency...")
idf = dict()
for _, term_dict in tf.items():
    for term, _ in term_dict.items():
        if term in idf:
            idf[term] += 1
        else:
            idf[term] = 1

print("Saving files...")
with open("yelp_reviews_tfidf.pkl", 'wb') as file:
    pickle.dump((tf, idf), file)