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
4. Implement text retrieval algorithms that calculate a similarity score by comparing the combined user reviews for each restaurant to the user query. We use three different algorithms to compute the similarity score. Next, incorporate the average/weighted average sentiment for each restaurant and calculate a weighted score by weighing both the similarity score and average/weighted average sentiment score. Lastly, sort the restaurants from highest to lowest score and recommend the restaurants with the highest score.
5. Create an interactive dashboard to host our restaurant recommendation system. This dashboard will enable users to input the type of food they want into a query and select other parameters. The system will then output a ranked list of the top (10/25/50) restaurants based on their preferences.
