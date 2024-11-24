# CS410 Course Project: Restaurant Recommendation System in Philadelphia

This repository contains all of the code and information used to build a restaurant recommendation system for restaurants in Philadelphia for our CS410 Course Project. 

## Project Overview
### Project Description
The main goal of this project is to develop a restaurant recommendation/ranking system that allows a user to input the type of food they want into a query, with the system returning a ranking of the top restaurants based on the user's specification. This restaurant recommendation system is purely based on user reviews of restaurants (text data) without incorporating restaurant characteristics such as price range, cuisine type, meal type, or venue. Including these restaurant characteristics would result in a simple filter on these characteristics and selecting the restaurants that match the most characteristics specified by the user, which would be relatively trivial. Our recommendation system relies on both text retrieval, based on the similarity of user reviews to the query, and the sentiment of user reviews to rank restaurants according to a user's query. We filter our sample of restaurants to the top 200 restaurants in Philadelphia with the most reviews. In total, we incorporate approximately 90,000 user reviews from these top 200 restaurants into our recommendation system. 

### Motivation
Each year, there are many tourists that travel to Philadelphia to see iconic venues such as the Independence Hall and the Liberty Bell. In addition to its iconic sites, Philadelphia is also known to be one of the best cities for food in the U.S. (ranked #6 according to [U.S. News](https://travel.usnews.com/rankings/best-foodie-destinations-in-the-usa/)) and many tourists also travel to Philadelphia to experience the great food and diverse cuisines. However, with so many great restaurants to choose from, they often struggle to decide where to dine. Our restaurant recommendation system attempts to provide reliable recommendations to Philadelphia vistors seeking restaurants that align with their specified preferences and allow them to make a better-informed decision on where to dine.

## Project Outline
Our approach and steps are outlined as follows:

1. Construct a dataset that includes specific restaurant characteristics such as location, price range, hours of operation, cuisine type, and many other characteristics for the top 200 restaurants in Philadelphia. Additionally, construct another dataset that contains all user reviews for each of the top 200 restaurants as well as user characteristics such as the user rating for that particular restaurant, the average rating a user has given for all restaurants they reviewed, the total number of reviews a user has given, and the number of useful votes sent by the user. The restaurant characteristics dataset and user reviews dataset were constructed using the Yelp Open Dataset as the original data source.
   - For more information about the description of the Yelp Open Dataset and the JSON data files contained in the dataset, visit [Yelp's website](https://www.yelp.com/dataset) or the [Yelp Dataset page on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data). For the Yelp Dataset data documentation, visit [Yelp's website](https://www.yelp.com/dataset/documentation/main).

2. Perform data cleaning on both the restaurant characteristics dataset and user reviews dataset in order to prepare them for further analysis and the construction of our recommendation system.
3. Conduct sentiment analysis on all user reviews and compute a sentiment score for each user review. Next, compute the average sentiment score as well as a customized weighted average sentiment score for each restaurant. 4 different sentiment analysis techniques were utilized including 3 lexicon-based methods and 1 transformer-based method.
4. Build the restaurant recommendation system. Implement text retrieval algorithms that calculate a similarity score by comparing the combined user reviews for each restaurant to the user query. We use three different algorithms to compute the similarity score. Next, integrate the average/weighted average sentiment scores for each restaurant, as calculated by the four sentiment analysis techniques. Normalize these scores by computing their z-scores, then calculate the average of the z-scores from the four techniques. Finally, normalize the average z-scores. Afterward, combine the normalized average/weighted average sentiment scores with the similarity scores for each restaurant and calculate a weighted score by weighing both the similarity score and average/weighted average sentiment score. Lastly, sort the restaurants from highest to lowest score and recommend the restaurants with the highest score.
5. Create an interactive dashboard to host our restaurant recommendation system. This dashboard will enable users to input the type of food they want into a query and select other parameters. The system will then output a ranked list of the top (10/25/50) restaurants based on their preferences.

## Installation
### Prerequisites ###
- **Python version:** Python 3.8
- Git
- [Anaconda](https://docs.anaconda.com/anaconda/install/windows/)

### Conda Environment
We use a conda environment called `cs410_course_project` to keep our Python version and packages consistent across all machines. All of the required packages to run the scripts contained in `src` are listed in [requirements.txt](https://github.com/kingeddy11/CS410CourseProject/blob/main/requirements.txt). To install this environment, please use the following commands:

```bash
conda create -n cs410_course_project python=3.8
conda activate cs410_course_project
pip install -r requirements.txt
```

## Project Folder and File Overview
The project contains 2 main folders within this repository along with a link to an external Google Drive directory. This external directory stores the input data files, which were downloaded and extracted from Yelp, as well as the data files that are read in and outputted by our scripts in `src`. The details of each of these folders are highlighted down below.

### data ###
Here is a link to the [data](https://drive.google.com/drive/u/4/folders/1SBrhxD7Jwwzv--Ma-U25TZcBS5jFO_lj) folder. After cloning this repository, please move the `data` folder into the root directory of the repository. The `data` folder is organized into 4 subfolders.
- `yelp_raw_data` folder contains a zip file `yelp_raw_data.zip` downloaded from [Yelp Dataset page on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data) and stores the input data from the Yelp Dataset. Download and extract the contents of the zip file and place them into the `yelp_raw_data` folder.
- `data_cleaning` folder contains CSV files that were created from running the scripts in the `src/data_cleaning_scripts` directory in this repo. This folder stores all of the preliminary data cleaning files that were created prior to the sentiment analysis step.
- `sentiment_analysis` folder contains CSV files that were created from running the scripts in the `src/sentiment_analysis_scripts` directory in this repo. This folder stores all of the sentiment analysis files that were created prior to building the restaurant recommendation system.
- `restaurant_rankings_by_query` folder contains CSV files that were created from running the scripts in the `src/restaurant_ranking_scripts`. These files contain a list of restaurant rankings generated from an example input user query "Japanese sushi", sorted by weight score from highest to lowest. Each restaurant is accompanied by its corresponding weighted score, which is calculated by weighing both the similarity score generated from our text retrieval algorithm and the average or weighted average sentiment score generated by the different sentiment analysis techniques.

### src ###
The `src` folder contains all of the programming scripts used in our project. The `src` folder is organized into 3 subfolders.
- `data_cleaning_scripts` contains all of the preliminary data cleaning scripts used to clean and prepare the data for sentiment analysis and building the recommendation system.
- `sentiment_analysis_scripts` contains the 2 sentiment analysis scripts used to compute sentiment scores for each user review and calculate the average sentiment score as well as a customized weighted average sentiment score for each restaurant using 3 lexicon-based methods ([`yelp_user_reviews_lexicon_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_lexicon_SA.ipynb)) and 1 transformer-based method ([`yelp_user_reviews_BERT_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_BERT_SA.py)).
- `restaurant_ranking_scripts` contains all of the scripts used to build the restaurant recommendation system using 3 different algorithms: word2vec embedding, pivoted length normalization, and BM25.

### figures ###
The `figures` folder contains PNG files that visualize the distribution of the sentiment scores computed from each of the 4 sentiment analysis techniques (vaderSentiment, TextBlob, SentiWordNet, BERT) for each of the ~90,000 user reviews. These PNG files were outputted from the scripts in the `src/sentiment_analysis_scripts` directory and the files for the 3 lexicon-based methods (VADER, TextBlob, SentiWordNet) were generated from [`yelp_user_reviews_lexicon_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_lexicon_SA.ipynb) while the file for BERT was generated from [`yelp_user_reviews_BERT_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_BERT_SA.py).

## Project Implementation and Steps
### 1. Data Cleaning ### 
Fill in

### 2. Sentiment Analysis ###
After cleaning the data and constructing the final version of the restaurant characteristics dataset `data/data_cleaning/yelp_restaurants_Phila_final.csv` and the user reviews dataset `yelp_reviews_users_Phila_final.csv` for the top 200 restaurants in Philadelphia, we implemented various sentiment analysis techniques to compute a sentiment score for each user review. 

#### Lexicon-Based Methods ####
The ([`yelp_user_reviews_lexicon_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_lexicon_SA.ipynb)) uses 3 different lexicon-based methods (VADER, TextBlob, SentiWordNet) to compute a sentiment score for each user review. Lexicon-based methods rely on a predefined list of words and phrases with associated sentiment scores (positive, negative, and neutral) and aggregates the sentiment scores of these individual words or phrases to compute the overall sentiment score of the text. VADER (Valence Aware Dictionary and Sentiment Reasoner) is a rule-based model that is optimized for social media data and uses a pre-defined lexicon trained on social media data with intensity values for words and various rules for punctuation, capitalization, and negations to adjust the overall sentiment of a text. TextBlob is a model that combines word-level polarity and subjectivity scores from a predefined lexicon trained on a large corpus and averages them across the text to compute the overall sentiment of a text. SentiWordNet is a lexical resource derived from WordNet (a large lexical database of English words) that assigns sentiment scores (positive, negative, and neutral) to synsets (a set of synonyms/related words) and averages the sentiment scores across the text (normalizing by word count) to compute the overall sentiment of a text. 

Prior to computing the sentiment score, the text in each review is cleaned by converting all words to lowercase, replacing instances of multiple whitespaces with single whitespaces, removing leading and trailing whitespaces, removing emojis, removing URLs, and removing emails. The 3 lexicon-based methods are then each individually applied to the cleaned review to compute a sentiment score (in the range [-1, 1]) for each review and method. We then visualize the distribution of sentiment scores calculated by each method for each user review and examine the average sentiment score for each method across different user ratings to verify that lower user ratings correspond to lower sentiment scores and higher ratings correspond to higher sentiment scores. We then compute various other statistics such as the word count for each review and a sentiment label (positive, negative, neutral) based on sentiment scores computed by the 3 lexicon-based methods. The sentiment labels are based on specific thresholds for its sentiment scores, with positive scores above a positive threshold, neutral scores within a defined range around zero, and negative scores below the lower threshold. Next, we calculate the average sentiment score as well as a customized weighted average sentiment score for each restaurant. 

The weighted average sentiment score for each restaurant is calculated by first normalizing selected contributing factors (all factors are in the range [0, 1]) such as the total number of reviews a user has posted, the usefulness of the review (number of votes that other users have cast to indicate if the review was helpful), the review word count, and the user rating of the restaurant (factoring in the user's average rating for all restaurants they reviewed). These normalized factors are then combined into a sentiment score weight using self-selected weights (0.2 for user review count, 0.35 for useful user review count, 0.05 for review word count, 0.4 for user rating) that represent their relative importance. These weights can be changed. Next, for each business, the weighted average sentiment score is calculated separately for each lexicon-based method by multiplying each review's sentiment score by its sentiment score weight, summing these products across all reviews for that business, and dividing the sum of all sentiment score weights. The weighted average sentiment score gives higher influence to reviews with greater relative importance.

##### Outputs: #####
Two CSV files are outputted by ([`yelp_user_reviews_lexicon_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_lexicon_SA.ipynb)).
1. **data/sentiment_analysis/yelp_reviews_lexicon_sentiment_analysis_Phila.csv**: This file contains the processed user reviews with sentiment analysis results for each review based on the 3 lexicon-based methods. It includes the original columns in `yelp_reviews_users_Phila_final.csv`, as well as the review_cleaned and review_word_count columns, in addition to the sentiment score columns and their corresponding sentiment labels. This file is not used in the construction of the restaurant recommendation system and is only used to sanity-check the sentiment scores produced by the 3 lexicon-based methods. 
2. **data/sentiment_analysis/yelp_restaurants_lexicon_sentiment_Phila.csv**: This file contains the computed average and weighted average sentiment scores as well as the number of positive, negative, and neutral reviews for each restaurant and lexicon-based method based on the user reviews. This file is used in the construction of the restaurant recommendation system.

#### BERT Transformer Method ####
In addition to the 3 lexicon-based methods, we also used BERT (Bidirectional Encoder Representations from Transformers) model, a transformer-based method, to compute sentiment scores for user reviews which can be found in ([`yelp_user_reviews_BERT_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_BERT_SA.py)). The BERT model leverages a bidirectional attention mechanism to capture context and semantic meanings, enabling it to classify sentiment based on pre-trained and fine-tuned representations of input sentences or documents. We specifically use the [Hugging Face: bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) to compute a sentiment label and confidence score for each review. In order to calculate the sentiment score for each review, sentiment labels are then mapped to numerical sentiment scores (5 stars = 1.0, 4 stars = 0.5, 3 stars = 0.0, 2 stars = -0.5, 1 star = -1.0) and the numerical sentiment scores are multiplied by the confidence score. The sentiment scores range from [-1, 1]. We follow the same ideology and steps as described in the implementation of the lexicon-based methods but instead using the BERT model.

##### Outputs: #####
Two CSV files are outputted by ([`yelp_user_reviews_BERT_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_BERT_SA.py)).
1. **data/sentiment_analysis/yelp_reviews_bert_sentiment_analysis_Phila.csv**: This file contains the processed user reviews with sentiment analysis results for each review. It includes the business_id, review, review_cleaned, and review_word_count columns along with the sentiment score column and respective sentiment label. This file is not used in the construction of the restaurant recommendation system and is only used to sanity-check the sentiment score produced by the BERT model. 
2. **data/sentiment_analysis/yelp_restaurants_bert_sentiment_Phila.csv**: This file contains the computed average and weighted average sentiment scores as well as the number of positive, negative, and neutral reviews for each restaurant. This file is used in the construction of the restaurant recommendation system. 
