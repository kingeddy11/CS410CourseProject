import pickle
from nltk.corpus import stopwords
import pandas as pd
from collections import defaultdict, Counter
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag

print("Loading data...")
df = pd.read_csv('../../data/data_cleaning/yelp_reviews_users_Phila_final.csv')
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
with open("yelp_reviews_preprocess_word2vec.pkl", 'wb') as file:
    pickle.dump((tf, idf), file)