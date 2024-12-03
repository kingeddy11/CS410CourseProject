# CS410 Course Project: Restaurant Recommendation System in Philadelphia

This repository contains all of the code and information used to build a restaurant recommendation system for restaurants in Philadelphia for our CS410 Course Project. 
## How to use Software (Restaurant Recommendation Dashboard)
Dashboard site: (DASHBOARD SITE)
1.	Start Your Search: Open the dashboard and locate the sidebar for search settings.
2.	Enter Your Query: Type what you're looking for in the search box, such as "best Italian food."
3.	Select a Search Method: Choose from BM25, PLN, or Word2Vec to define how the search algorithm ranks results.
4.	Adjust Weighting Preferences: Use the Similarity Score Weight slider to control how much emphasis is placed on the relevance of text matches versus sentiment analysis. The sentiment score is the complement of the weighted score (i.e. 1 - similarity_score). The sentiment score and similarity score are used to compute the weighted score. 
5.	Refine Results with Length Normalization: If you’re using PLN, tweak the Length Normalizer Weight slider to account for variations in document length.
6.	Choose Sentiment Scoring: Check the box to apply Weighted Average Sentiment Scoring for more nuanced results, or leave it unchecked to use simple averages.
7.	Set the Number of Results: Decide how many restaurants you’d like to see—10, 25, or 50.
8.	Find Restaurants: Click the button to generate a list of recommended restaurants based on your settings.
9.	Review Recommendations: The results will be displayed in a table showing the restaurant names, business IDs, and scores. Adjust your settings to explore different rankings.

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
4. Build the restaurant recommendation system. Implement text retrieval algorithms that calculate a similarity score by comparing the combined user reviews for each restaurant to the user query. We use three different algorithms to compute the similarity score. Next, integrate the average/weighted average sentiment scores for each restaurant, as calculated by the four sentiment analysis techniques. Normalize these scores by computing their z-scores, then calculate the average of the z-scores from the four techniques. Finally, normalize the average z-scores. Afterwards, combine the normalized average/weighted average sentiment scores with the similarity scores for each restaurant and calculate a weighted score by weighing both the similarity score and average/weighted average sentiment score. Lastly, sort the restaurants from highest to lowest score and recommend the restaurants with the highest scores.
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
- `dashboard_creation_scripts` contains all of the scripts used to create our dashboard which is used to host our restaurant recommendation system using 3 different algorithms: word2vec embedding, pivoted length normalization, and BM25.

### models ###
Make sure to create a `models` folder in the root directory of this repo. For our word2vec model, make sure to download [`GoogleNews-vectors-negative300.bin`](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors) and place the downloaded bin file in the newly created `models` folder. We are not able to place the downloaded bin file and upload to Github since the model exceeds the Github file size limit.

### figures ###
The `figures` folder contains PNG files that visualize the distribution of the sentiment scores computed from each of the 4 sentiment analysis techniques (vaderSentiment, TextBlob, SentiWordNet, BERT) for each of the ~90,000 user reviews. These PNG files were outputted from the scripts in the `src/sentiment_analysis_scripts` directory and the files for the 3 lexicon-based methods (VADER, TextBlob, SentiWordNet) were generated from [`yelp_user_reviews_lexicon_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_lexicon_SA.ipynb) while the file for BERT was generated from [`yelp_user_reviews_BERT_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_BERT_SA.py).

## Project Implementation and Steps
The project implementation and steps are listed and executed in chronological order with each step/part building upon the previous one.

### 1. Data Cleaning ### 
The data preprocessing and data cleaning is split into 3 parts. Part 1 focuses on loading in the raw data files downloaded and extracted from [Yelp Dataset page on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data) and constructing a dataset of restaurant characteristics and a dataset of user reviews for the top 200 restaurants in Philadelphia. Part 2 cleans the restaurant characteristics dataset and user reviews datasets outputted in Part 1 by using Excel to address encoding issues such as removing "=-" in a few observations and using OpenRefine to further clean some of the columns in the user reviews dataset. Part 3 further cleans the restaurant characteristics dataset and the user reviews dataset by unnesting columns that are dictionary-like objects such as `Ambience` and `GoodForMeal`, cleaning columns that had "u'" or unnecessary quotes, converting column types to the proper data type, and selecting the columns we will be using in our restaurant recommendation system. After Part 3, two final, fully-cleaned CSV files (restaurant characteristics dataset and user reviews dataset) are outputted in preparation to use for sentiment analysis and to build our restaurant recommendation system.

#### Part 1 ####
([`yelp_data_cleaning_1.ipynb`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/data_cleaning_scripts/yelp_data_cleaning_1.ipynb)) contains all of the data cleaning performed in Part 1. Prior to Part 1, the `yelp_raw_data.zip` file is downloaded from [Yelp Dataset page on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data) and the contents of the zip file are extracted creating 5 separate JSON files (`yelp_academic_dataset_business.json`, `yelp_academic_dataset_checkin.json`, `yelp_academic_dataset_review.json`, `yelp_academic_dataset_tip.json`, `yelp_academic_dataset_user.json`) which are placed in the `data/yelp_raw_data` folder. Due to the large size of these JSON files, we used the [DuckDB API](https://duckdb.org/docs/api/python/data_ingestion#json-files) to load in the reviews dataset (`yelp_academic_dataset_review.json`), users dataset (`yelp_academic_dataset_user.json`), and business dataset (`yelp_academic_dataset_business.json`). The reviews dataset contains full review text data including the user who wrote the review and the restaurant the review was written for. The users dataset contains all of the metadata associated with a user. The business dataset contains information about a particular business such as its location, price range, hours of operation, categories, and many other characteristics. 

After loading in these datasets, the business dataset was filtered for only businesses in Philadelphia, Pennsylvania after cleaning the `city` column. We then filtered for only restaurants based on the `categories` column and grabbed the top 200 open restaurants with the most reviews to create our restaurant characteristics dataset. The restaurant characteristics dataset was further cleaned by unnesting the key-value pairs in the `attributes` column into their own separate columns and unnesting the `hours` column to create binary variables indicating whether a restaurant was open on each day of the week. Next, we merged together the business dataset (filtered for the top 200 restaurants in Philadelphia), reviews dataset, and the users dataset and grabbed the relevant columns from each dataset (`business_id`, `restaurant_name`, `review`, `user_review_count`, etc) to construct our user reviews dataset.

##### Outputs: #####
Two CSV files are outputted by ([`yelp_data_cleaning_1.ipynb`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/data_cleaning_scripts/yelp_data_cleaning_1.ipynb)).
1. **data/data_cleaning/yelp_restaurants_Phila.csv**: This file contains the first version of the restaurant characteristics dataset.
2. **data/data_cleaning/yelp_reviews_users_Phila.csv**: This file contains the first version of the user reviews dataset.

#### Part 2 ####
After creating the initial version of the restaurant characteristics dataset and the user reviews dataset CSV files, we used Excel to further clean some of the observations in each dataset. As mentioned previously, there were a few observations that had a `#NAME?` error in both CSV files which were caused by "=-" in a few observations. To resolve this and accurately retrieve the actual values, we removed these occurrences. Afterwards, we use OpenRefine to clean several columns in the user reviews dataset, including trimming leading and trailing whitespace/collapsing consecutive whitespace in the `review` column and converting other columns to the appropriate data types. ([`yelp_reviews_users_cleaning_openrefine.json`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/data_cleaning_scripts/yelp_reviews_users_cleaning_openrefine.json) contains the specific data cleaning steps using OpenRefine.

##### Outputs: #####
Two CSV files are saved after Part 2.
1. **data/data_cleaning/yelp_restaurants_Phila_cleaned.csv**: This file contains the second version of the restaurant characteristics dataset.
2. **data/data_cleaning/yelp_reviews_users_Phila_cleaned.csv**: This file contains the second version of the user reviews dataset.

#### Part 3 ####
([`yelp_data_cleaning_2.ipynb`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/data_cleaning_scripts/yelp_data_cleaning_2.ipynb)) contains all of the data cleaning performed in Part 3. We start off by loading in the second versions of the restaurant characteristics dataset and user reviews dataset. We then drop all columns not used in our analysis or the construction of the restaurant recommendation system from both datasets. We then clean the `NoiseLevel`, `Alcohol`, and `RestaurantsAttire` columns by removing instances with "u'" or unnecessary quotes from observations in these columns. Next, we clean the `Ambience` and `GoodForMeal` columns and unnest the key-value pairs contained in these columns into their own separate columns. Afterwards, we standardize certain columns by replacing None values with NA values and convert specific columns to the appropriate data type. Lastly, we output the final versions of the restaurant characteristics dataset and user reviews dataset which are used in the next steps.

##### Outputs: #####
Two CSV files are outputted by ([`yelp_data_cleaning_2.ipynb`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/data_cleaning_scripts/yelp_data_cleaning_2.ipynb)).
1. **data/data_cleaning/yelp_restaurants_Phila_final.csv**: This file contains the final version of the restaurant characteristics dataset.
2. **data/data_cleaning/yelp_reviews_users_Phila_final.csv**: This file contains the final version of the user reviews dataset.

### 2. Sentiment Analysis ###
After cleaning the data and constructing the final version of the restaurant characteristics dataset `data/data_cleaning/yelp_restaurants_Phila_final.csv` and the user reviews dataset `yelp_reviews_users_Phila_final.csv` for the top 200 restaurants in Philadelphia, we implemented various sentiment analysis techniques to compute a sentiment score for each user review. 

#### Lexicon-Based Methods ####
([`yelp_user_reviews_lexicon_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_lexicon_SA.ipynb)) uses 3 different lexicon-based methods (VADER, TextBlob, SentiWordNet) to compute a sentiment score for each user review. Lexicon-based methods rely on a predefined list of words and phrases with associated sentiment scores (positive, negative, and neutral) and aggregates the sentiment scores of these individual words or phrases to compute the overall sentiment score of the text. VADER (Valence Aware Dictionary and Sentiment Reasoner) is a rule-based model that is optimized for social media data and uses a pre-defined lexicon trained on social media data with intensity values for words and various rules for punctuation, capitalization, and negations to adjust the overall sentiment of a text. TextBlob is a model that combines word-level polarity and subjectivity scores from a predefined lexicon trained on a large corpus and averages them across the text to compute the overall sentiment of a text. SentiWordNet is a lexical resource derived from WordNet (a large lexical database of English words) that assigns sentiment scores (positive, negative, and neutral) to synsets (a set of synonyms/related words) and averages the sentiment scores across the text (normalizing by word count) to compute the overall sentiment of a text. 

Prior to computing the sentiment score, the text in each review is cleaned by converting all words to lowercase, replacing instances of multiple whitespaces with single whitespaces, removing leading and trailing whitespaces, removing emojis, removing URLs, and removing emails. The 3 lexicon-based methods are then each individually applied to the cleaned review to compute a sentiment score (in the range [-1, 1]) for each review and method. We then visualize the distribution of sentiment scores calculated by each method for each user review and examine the average sentiment score for each method across different user ratings to verify that lower user ratings correspond to lower sentiment scores and higher ratings correspond to higher sentiment scores. We then compute various other statistics such as the word count for each review and a sentiment label (positive, negative, neutral) based on sentiment scores computed by the 3 lexicon-based methods. The sentiment labels are based on specific thresholds for its sentiment scores, with positive scores above a positive threshold, neutral scores within a defined range around zero, and negative scores below the lower threshold. Next, we calculate the average sentiment score as well as a customized weighted average sentiment score for each restaurant. 

The average sentiment score is a simple arithmetic mean of the sentiment scores computed from all reviews for a particular restaurant. The weighted average sentiment score for each restaurant is calculated by first normalizing selected contributing factors (all factors are in the range [0, 1]) such as the total number of reviews a user has posted, the usefulness of the review (number of votes that other users have cast to indicate if the review was helpful), the review word count, and the user rating of the restaurant (factoring in the user's average rating for all restaurants they reviewed). These normalized factors are then combined into a sentiment score weight using self-selected weights (0.2 for user review count, 0.35 for useful user review count, 0.05 for review word count, 0.4 for user rating) that represent their relative importance. These weights can be changed. Next, for each business, the weighted average sentiment score is calculated separately for each lexicon-based method by multiplying each review's sentiment score by its sentiment score weight, summing these products across all reviews for that business, and dividing the sum of all sentiment score weights. The weighted average sentiment score gives higher influence to reviews with greater relative importance.

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

### 3. Ranking / Search System ###
For the recommendation system, we need a method to rank search results for restaurants based on the review content. We tried three separate methods for ranking, with varying degrees of effectiveness.

#### Word2Vec method ####
Word2Vec is a method of mapping words to high-dimensional vectors such that semantic meaning is retained in the pairwise distances of vectors. For example, "dog" and "cat", both being pets, may be closer in the vector space than "dog" and "elephant". Our initial intuition is that each resturant can be mapped to a high-dimensional vector by computing a weighted average of the (precomputed) word2vec vectors for each word for every review. We should then be able to search for restaurant which feature reviews that are semantically closest to a given query. In [`word2vec_preprocess.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/word2vec_preprocess.py), we lemmatize every word for every review of a given restaurant, filtering out stop words, then aggregate TF and IDF values. In [`word2vec_construct_vector.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/word2vec_construct_vector.py), for each restaurant, we construct a TF-IDF weighted average vector representation, using the pretrained vector model [`GoogleNews-vectors-negative300`](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors). Make sure to download [`GoogleNews-vectors-negative300.bin`](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors) and create a `models` folder in the root directory of this repo and place the downloaded bin file in the newly created `models` folder. Finally, we apply the same process to the given query (e.g. "Japanese sushi") in  [`word2vec_search.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/word2vec_search.py), to compare vector similarity between the query and each restaurant, resulting in a ranking. 

##### Intermediate files #####
TF-IDF count: [`yelp_reviews_preprocess_word2vec.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/yelp_reviews_preprocess_word2vec.pkl)

Document vectors: [`yelp_reviews_doc_vectors_word2vec.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/yelp_reviews_doc_vectors_word2vec.pkl)

#### Pivoted Length Normalization method ####
We explored another method using TF-IDF ranking, this time without using the Word2Vec-based similarity approach but simply implementing TF-IDF with pivoted length normalization. The 3-step process is similar to the word2vec, with an initial aggregation of term frequency / document frequency ([`piv_len_norm_preprocess.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/piv_len_norm_preprocess.py)), then a process to map each restaraunt to its vector representation ([`piv_len_norm_construct_vector.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/piv_len_norm_construct_vector.py)), and finally a ranking based on these vectors ([`piv_len_norm_search.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/piv_len_norm_search.py)). We noticed that, in most cases, this method tends to outperform the Word2Vec-based approach.

##### Intermediate files #####
TF-IDF count: [`yelp_reviews_preprocess_pln.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/yelp_reviews_preprocess_pln.pkl)

Document vectors: [`yelp_reviews_doc_vectors_pln.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/yelp_reviews_doc_vectors_pln.pkl)

#### BM25 ####
BM25 is a ranking algorithm primarily based on TF-IDF. First developed in the 1970s, it continues to see usage due to its robustness as a retrieval method for text documents. While our TF-IDF implemention is very similar to it, BM25 offers two further optimizations. First, it introduces the concept of frequency saturation, allow repeated occurrences of the same word in a document to have a diminishing effect on the score. Second, the frequency saturation and length normalization components can be tuned with the free parameters `k1` and `b`. We use the [`rank-bm25`](https://pypi.org/project/rank-bm25/) python library to implement this search system. In [`bm25_preprocess.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/bm25_preprocess.py), we construct the BM25Okapi object with the stemmed corpus, using parameters of b = 0 and k1 = 2.  In [`bm25_search.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/bm25_search.py), we execute the search based on the input query and return a ranking of the restaurants.

##### Intermediate files #####
BM25 pre-processed documents: [`yelp_reviews_preprocess_bm25.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/restaurant_ranking_scripts/yelp_reviews_preprocess_bm25.pkl)

#### Final Weighted Score (Ranking Score) Calculation ####
After computing the similarity scores using the 3 different algorithms (word2vec, pivoted length normalization, and BM25) in the "_search.py" scripts, we integrate the average/weighted average sentiment scores for each restaurant as calculated by the four sentiment analysis techniques (VADER, TextBlob, SentiWordNet, BERT). We then normalize these scores by computing their z-scores and calculate the average of the z-scores from the four techniques. Finally, normalize the average z-scores. Afterwards, we combine the normalized average/weighted average sentiment scores with the similarity scores for each restaurant and calculate an overall weighted score (`weighted_score`) by weighing both the similarity score and average/weighted average sentiment score. Lastly, we sort the restaurants from highest to lowest score and recommend the restaurants with the highest scores. We allow the user to specify the weight they want to place on the similarity scores (in the range [0, 1]) and the sentiment scores weight is calculated as 1 - similarity scores weight. We also allow the user to select whether they want to use the average sentiment score or weighted average sentiment score when computing the overall weighted score used in the final ranking of restaurants.

##### Output Using Example Queries: #####
Two CSV files (a ranking based on the weighted average sentiment score and a ranking based on the average sentiment score) are outputted using each of the 3 methods displaying the results of the 200 restaurants ranked (highest `weighted_score` to lowest) based on an example query "Japanese sushi". `business_id`, `restaurant_name`, `weighted_score`, similarity score columns, and sentiment score columns are shown for each of the 200 restaurants.
1. **data/restaurant_rankings_by_query/restaurant_ranking_word2vec_w_avg_sentiment_japanese sushi.csv**: This file contains the rankings generated using the word2vec method for computing the similarity score, along with the weighted average sentiment scores that contribute to the overall weighted score used for ranking. 
2. **data/restaurant_rankings_by_query/restaurant_ranking_word2vec_avg_sentiment_japanese sushi.csv**: This file contains the rankings generated using the word2vec method for computing the similarity score, along with the average sentiment scores that contribute to the overall weighted score used for ranking.
3. **data/restaurant_rankings_by_query/restaurant_ranking_pln_w_avg_sentiment_japanese sushi.csv**: This file contains the rankings generated using the pivoted length normalization method for computing the similarity score, along with the weighted average sentiment scores that contribute to the overall weighted score used for ranking. 
4. **data/restaurant_rankings_by_query/restaurant_ranking_pln_avg_sentiment_japanese sushi.csv**: This file contains the rankings generated using the pivoted length normalization method for computing the similarity score, along with the average sentiment scores that contribute to the overall weighted score used for ranking.
5. **data/restaurant_rankings_by_query/restaurant_ranking_bm25_w_avg_sentiment_japanese sushi.csv**: This file contains the rankings generated using the bm25 method for computing the similarity score, along with the weighted average sentiment scores that contribute to the overall weighted score used for ranking. 
6. **data/restaurant_rankings_by_query/restaurant_ranking_bm25_avg_sentiment_japanese sushi.csv**: This file contains the rankings generated using the bm25 method for computing the similarity score, along with the average sentiment scores that contribute to the overall weighted score used for ranking.

### 4. User Interface
The Restaurant Recommendation System is a software designed to help users discover the best dining options in Philadelphia by combining their preferences with sentiment analysis of Yelp reviews. Key features include a Search Box for entering queries (e.g., "best brunch") and selecting a retrieval method such as BM25, Pivoted Length Normalization (PLN), or Word2Vec. Users can customize their experience with Parameter Options, choosing between weighted or simple average sentiment scores, and adjust sentiment influence using a Sentiment Weight Slider ranging from 0 to 1. The software provides Sentiment-Enhanced Rankings, which combine traditional similarity scores with review sentiment, and displays Real-Time Results with ranked restaurants, business IDs, and final scores. Users can also select how many results to display (10, 25, or 50) using Scalable Options. Advanced text processing techniques like tokenization, stemming, and stopword removal, along with lexicon-based and BERT-based sentiment analysis of ~90,000 user reviews for 208 restaurants, ensure accurate, user-focused, and data-driven recommendations.
The user interface contains two scripts: 
1. search_restaurants.py
This script handles the search and ranking process for the recommendation system. It includes functions to preprocess user queries by tokenizing, lemmatizing, and removing stopwords (preprocess_query). The scoring methods for BM25, Pivoted Length Normalization (PLN), and Word2Vec are implemented in bm25_scoring, pln_scoring, and w2c_scoring, respectively. The main function, search_restaurants, combines these retrieval methods with sentiment analysis to rank restaurants based on user input. Sentiment scores are factored into the rankings using a weighted system.
2. streamlit_dashboard.py
This script is used to create the interactive dashboard with Streamlit. It provides input fields for user queries, options to select retrieval methods (BM25, PLN, or Word2Vec), and sliders to adjust sentiment weight and normalization parameters. Users can also select how many results to display. The script connects to search_restaurants.py to process the queries and display ranked restaurant recommendations in real-time.

