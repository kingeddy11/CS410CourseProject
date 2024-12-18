# -*- coding: utf-8 -*-
"""sentiment BERT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nDplXYkgh0z_8Trf0-_OaknzkiuA_dNz
"""

import nltk # Import the nltk module
import os
import re
import emoji
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import nltk
import torch
from datasets import Dataset
from nltk.tokenize import word_tokenize
text = "This is a test sentence."
tokens = word_tokenize(text)
print(tokens)


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

def cleanText(text):
    """Clean the review text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = emoji.demojize(text)
    text = re.sub(r'\s*http(s?)\S+|\s*www\S+', '', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}', '', text)
    return text

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load model and tokenizer explicitly
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(f"cuda:{device}" if device == 0 else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the pipeline
SENTIMENT_ANALYZER = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Map labels to sentiment scores
LABEL_TO_SCORE = {
    "5 stars": 1.0,
    "4 stars": 0.5,
    "3 stars": 0.0,
    "2 stars": -0.5,
    "1 star": -1.0
}

def analyze_sentiments(df):
    """Analyze sentiments for each review and compute sentiment scores."""
    try:
        print("Step 3: Analyzing sentiments...")

        # Convert the DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df[['review_cleaned']])

        # Define a function to map the sentiment analysis over the dataset
        def predict_sentiment(batch):
            predictions = SENTIMENT_ANALYZER(batch["review_cleaned"], truncation=True, max_length=512)
            scores = [
                LABEL_TO_SCORE[pred["label"]] * pred["score"]
                for pred in predictions
            ]
            return {"sentiment_score": scores}

        # Apply to the dataset
        dataset = dataset.map(
            predict_sentiment,
            batched=True,
            batch_size=64
        )

        # Extract and return as a list
        sentiment_scores = dataset["sentiment_score"]
        print("Sentiment analysis completed.")
        return sentiment_scores
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return []

def main():


    INPUT_FILE_PATH = "../../data/data_cleaning/yelp_reviews_users_Phila_final.csv"
    REVIEW_OUTPUT_FILE_PATH = '../../data/sentiment_analysis/yelp_reviews_bert_sentiment_analysis_Phila.csv'
    BUSINESS_OUTPUT_FILE_PATH = '../../data/sentiment_analysis/yelp_restaurants_bert_sentiment_Phila.csv'

    # Step 1: Load data
    print("Step 1: Loading data...")
    df = pd.read_csv(INPUT_FILE_PATH, engine="python")
    print(f"Data loaded successfully. {len(df)} rows and {len(df.columns)} columns.")

    # Step 2: Clean text
    print("Step 2: Cleaning review text...")
    df["review_cleaned"] = df["review"].apply(cleanText)
    print("Review cleaning completed.")

    # Step 3: Analyze sentiments
    print("Step 3: Analyzing sentiments...")
    df["sentiment_score"] = analyze_sentiments(df)
    print("Sentiment analysis completed.")

   # Add sentiment label for each row
    print("Adding sentiment labels...")
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda x: "positive" if x > 0 else "neutral" if x == 0 else "negative"
    )

  #  Plot distribution of sentiment scores
    print("Plotting sentiment score distribution...")
    plt.figure(figsize=(10, 6))
    plt.hist(df["sentiment_score"], bins=30, color="skyblue", edgecolor="black")
    plt.title("Distribution of Sentiment Scores")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

     # Normalize variables/factors
    print("Step 4: Normalizing variables/factors...")
    df["review_word_count"] = df["review_cleaned"].apply(
        lambda x: len([word for word in word_tokenize(str(x)) if word.isalnum()])
    )

    # Normalize user_review_count for each business_id
    df["norm_user_review_count"] = df.groupby("business_id")["user_review_count"].transform(lambda x: x / x.max())

    # Normalize useful_user_review_count for each business_id
    df["norm_useful_user_review_count"] = df.groupby("business_id")["useful_user_review_count"].transform(lambda x: x / x.max())

    # Normalize review_word_count for each business_id
    df["norm_review_word_count"] = df.groupby("business_id")["review_word_count"].transform(lambda x: x / x.max())

    # Normalize user rating
    scaler = MinMaxScaler()
    df["norm_user_rating"] = scaler.fit_transform(
        (df["user_rating"] - df["user_average_rating"]).values.reshape(-1, 1)
    )
    print("Normalization completed.")



    # Step 5: Calculate weighted sentiment score
    print("Step 8: Calculating weighted sentiment score...")
    weight_user_review_count = 0.2
    weight_useful_review_count = 0.35
    weight_word_count = 0.05
    weight_sentiment_score = 0.4

    df["weighted_sentiment_score"] = (
        weight_user_review_count * df["norm_user_review_count"] +
        weight_useful_review_count * df["norm_useful_user_review_count"] +
        weight_word_count * df["norm_review_word_count"] +
        weight_sentiment_score * df["sentiment_score"]
    )

    # Save review-level data
    print("Saving review-level data...")
    review_columns = [
        "business_id", "review", "review_cleaned", "review_word_count",
        "sentiment_label", "sentiment_score", "weighted_sentiment_score"
    ]
    df[review_columns].to_csv(REVIEW_OUTPUT_FILE_PATH, index=False)
    print(f"Review-level data saved to {REVIEW_OUTPUT_FILE_PATH}")

     # Step 6: Aggregate sentiment scores at the business level
    print("Step 6: Aggregating sentiment scores...")
    sentiment_counts_per_business = df.groupby("business_id")["sentiment_label"].value_counts().unstack(fill_value=0)
    sentiment_counts_per_business = sentiment_counts_per_business.rename(
        columns={"positive": "positive_count", "neutral": "neutral_count", "negative": "negative_count"}
    )
    business_sentiment = df.groupby("business_id").agg(
        avg_sentiment_score=("sentiment_score", "mean"),
        avg_weighted_sentiment_score=("weighted_sentiment_score", "mean"),
    ).reset_index()
    business_data = business_sentiment.merge(sentiment_counts_per_business, on="business_id")

    # Save business-level data
    print("Saving business-level data...")
    business_data.to_csv(BUSINESS_OUTPUT_FILE_PATH, index=False)
    print(f"Business-level data saved to {BUSINESS_OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()

"""# New Section"""

