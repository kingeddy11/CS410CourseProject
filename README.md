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
The `figures` folder contains PNG files that visualize the distribution of the sentiment scores computed from each of the 4 sentiment analysis techniques (vaderSentiment, TextBlob, SentiWordNet, BERT) for each of the ~90,000 user reviews. These PNG files were outputted from the scripts in the `src/sentiment_analysis_scripts` directory and the files for the 3 lexicon-based methods (vaderSentiment, TextBlob, SentiWordNet) were generated from [`yelp_user_reviews_lexicon_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_lexicon_SA.ipynb) while the file for BERT was generated from [`yelp_user_reviews_BERT_SA.py`](https://github.com/kingeddy11/CS410CourseProject/blob/main/src/sentiment_analysis_scripts/yelp_user_reviews_BERT_SA.py).
