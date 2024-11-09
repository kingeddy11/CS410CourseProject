import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline


file_path = "(FILE PATH)"
df = pd.read_csv(file_path)

# Hugging face BERT model
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True)

# Sentiment
reviews = df['text'].tolist()
predictions = [sentiment_analyzer(review)[0] for review in reviews]


# Map to numerical values: -1, 0, 1
def map_sentiment(label):
    if label in ['1 star', '2 stars']:
        return -1
    elif label == '3 stars':
        return 0
    elif label in ['4 stars', '5 stars']:
        return 1
    return None


# Add to the DataFrame
df['predicted_sentiment'] = [map_sentiment(prediction['label']) for prediction in predictions]
df['confidence score'] = [prediction['score'] for prediction in predictions]

# Weighted sentiment score. Multiply predicted sentiment by confidence score.
# For neutral sentiment, apply log(1+confidence score) formula to avoid results being 0 while still center around 0.

df["Weighted sentiment score"] = np.where(
    df["predicted_sentiment"] == 0,
    np.log1p(df["confidence score"]),
    df["predicted_sentiment"] * df["confidence score"]  

# Compute the average weighted sentiment score for each restaurant
avg_weighted_score = df.groupby("name_business_reviews")["Weighted sentiment score"].mean().reset_index()
avg_score_dict = dict(zip(avg_weighted_score["name_business_reviews"], avg_weighted_score["Weighted sentiment score"]))

# Add the average score per restaurant to each row
df["Average score by restaurant"] = df["name_business_reviews"].map(avg_score_dict)

# Word counts for each review entry
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))


print(df[["text", "predicted_sentiment", "sentiment_score", "Weighted sentiment score", "Average score by restaurant", "word_count"]].head())

# Plotting the distribution of the weighted sentiment score
plt.figure(figsize=(10, 6))
plt.hist(df["Weighted sentiment score"], bins=30, color='blue', edgecolor='black')
plt.title("Distribution of Weighted Sentiment Score")
plt.xlabel("Weighted Sentiment Score")
plt.ylabel("Frequency")
plt.show()


output_path = "(FILE PATH)"
df.to_csv(output_path, index=False)

