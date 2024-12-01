import streamlit as st
st.set_page_config(page_title="Restaurant Finder in Philadelphia, PA", page_icon="🍽️")
import nltk
# nltk.data.path.append("/workspaces/movies-dataset-1/nltk_data")
# Download necessary datasets
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For WordNet Lemmatizer
import pandas as pd

import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
import os
from rank_bm25 import BM25Okapi

st.title("🍽️ Restaurant Finder in Philadelphia")
st.write(
    """
    Welcome to the Restaurant Finder app! Explore top restaurants in Philadelphia
    based on user preferences and sentiment analysis. Customize your search and
    find the perfect dining spot!
    """
)

# Sidebar Widgets for User Inputs
st.sidebar.header("Customize Your Search")

# Query Input
query = st.sidebar.text_input("Enter your search query", value="best Italian food")

# Algorithm Selection
method = st.sidebar.radio("Choose Search Method", ["bm25", "pln", "word2vec"])

# Similarity Score Weight and Length Normalizer weight (add footnote for sentiment score = 1 - sim score)
sim_score_wght = st.sidebar.slider(
    "Similarity Score Weight (α)", min_value=0.0, max_value=1.0, value=0.8
)
st.sidebar.markdown(
    "<small>*Note: sentiment score is the complement of the weighted score (i.e. 1 - similarity_score) and together these two scores are used to compute the weighted score</small>",
    unsafe_allow_html=True,
)

length_normalizer = st.sidebar.slider("Length Normalizer Weight (b)", min_value=0.0, max_value=1.0, value=0.9)

# Sentiment Score Weight Toggle
weighted_avg_sentiment = st.sidebar.checkbox(
    "Use Weighted Average Sentiment?", value=True
)
st.sidebar.markdown(
    "<small>*Note: If box is unchecked average sentiment is computed for each restaurant(i.e. business id)</small>",
    unsafe_allow_html=True,
)

# Number of Results to Display
top_n = st.sidebar.selectbox("Number of Results", [10, 25, 50])

# Import your search algorithm functions
from search_restaurants import search_restaurants

# Perform Search When the User Clicks the Button
if st.sidebar.button("Find Restaurants"):
    st.write(f"### Top {top_n} Restaurants for: '{query}' (Method: {method})")
    
    try:
        # Perform the search and get results                
        results = search_restaurants(
            query=query,
            method=method,
            num_results = top_n,            
            sim_score_wght=sim_score_wght,
            weighted_avg_sentiment=weighted_avg_sentiment,
            B = length_normalizer
        )
        st.dataframe(
            results.head(top_n).rename(
                columns={
                    "restaurant_name": "Restaurant Name",
                    "weighted_score": "Final Score",
                }
            ),
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
