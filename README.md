# NYC Taxi Trip Duration - BigData Analysis

> Implemented a big data analysis on the NYC taxi trip records using spark cluster. The aim was to analyze huge dataset using distributed cluster computing framework like Apache-spark and build a machine learning model with good accuracy to predict trip duration based on certain inputs.

# Local Environment Setup

```bash
pip install virtualenv            # install virtualenv

virtuenv venv                     # creates virtual environment

source venv/bin/activate          # activate virtualenv

pip install -r requirements.txt   # install require dependencies

Happy Coding... 😊
```

# I. Abstract

New York city is one of the major cities of the world, and taxi cabs are a popular form of transport for its residents. That is why in this project, we have explored ways of using machine learning models to predict the duration of taxi trips when given details such as start and end coordinates, number of passengers, month, quarter of the year, starting time and so on. A dataset from Kaggle containing the data of more than 0.5 million taxi trips was used in this project. This project aims to be particularly useful towards those, who want to check their trip time in NYC.

# II. Introduction

New York City (NYC) is one of the biggest cities in the world, with a population of around 18,000,000 people. Due to its large population, traffic congestion is one of the most critical problems in New York. In addition, taxi cabs are a popular choice of transport when traveling within this city. That is why in this project, we chose to predict the trip duration for a taxi in New York using the “NYC taxis trip duration dataset” which can be found <a href="https://github.com/adam-25/NYC-taxi-Trip-duration-BigData-Analysis/blob/main/Data/NYC%20Taxi%20Duration.csv" target="_blank">here</a> and so people can get an idea of how long their taxi trip will take and plan their trips beforehand. 

The current dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo coordinates, number of passengers, and several other variables. The dataset contains more than 510,000 rows. Since the dataset is mostly from a Kaggle challenge (where people compete to use the provided data to create the best visualizations, machine learning models and reports), there are a lot of machine learning models that were built using this dataset. However, we found out that the accuracy of these models is not very high, which means that there is a possibility of developing models that can achieve higher accuracy and reliability. Moreover, the current machine learning models were built using mostly Tensorflow and other CNN techniques, while we manipulated the data and create a model using SkLearn and PySpark.

In our solution, we have created several visualizations that show a relationship between trip duration and other variables, also we demonstrated how the mean passenger count, total trips, distance and duration vary on different weekdays. Also, we have created 3 machine learning models to either gain insights into the structure of the data or try to predict the duration of a trip. That includes regression models like linear regression, gradient boosted trees, and K-means clustering.

# III. Data Preprocessing

While preparing the data for the machine learning models, there were several operations performed on the dataset. We have changed the type of each column from string to either integer, double, string or timestamp according to its values. Also, we have calculated the distance between pickup latitude, longitude and dropoff latitude and longitude for all the trips. As we believe that taxi trips in New York City might not take more than 2 hours at any time we filtered only trips whose trip duration is between 3 minutes to 2 hours. In addition to that we believe that taxi rides in New York city may always be less than 60 kilometers as no taxi accepts rides as the destination is more than 60 kilometers far away. There are also several preprocessing has been done on the data before using it for the machine learning models.

<img width="902" alt="Screenshot 2023-01-18 at 11 27 15 PM" src="https://user-images.githubusercontent.com/92228157/213371082-3b847139-5b3e-4e9d-9acc-9b80532264a1.png">

<img width="981" alt="Screenshot 2023-01-18 at 11 38 41 PM" src="https://user-images.githubusercontent.com/92228157/213372876-aa1a338e-65b2-4e54-bc1c-f0bf05da54d7.png">

# IV. Data Visualization

To get good insights from the existing dataset, we have performed several data visualization. From the figure below, we can see that the total number of trips is higher on Friday and Saturday with the Average Passenger count higher on Sunday as well because people are more likely to go out with the family at the start of the weekend. While another grpah helps us to match our initial expectation that trip duration is lower on weekends as there would be less traffic because most of people are not working.

<img width="780" alt="Screenshot 2023-01-18 at 11 45 25 PM" src="https://user-images.githubusercontent.com/92228157/213373986-946fc068-cdaa-4a6a-a953-5f0068647283.png">

<img width="732" alt="Screenshot 2023-01-18 at 11 45 42 PM" src="https://user-images.githubusercontent.com/92228157/213374030-a9cf4162-f2ce-4afe-aafd-080e293bdf1c.png">

While predicting and getting detailed insights of the data regarding any duration of the trip the most important variable to consider is a distance and also most people rely on the distance of two places to predict their duration of the trip. We have plot the relationship between distance and trip duration. Also, grpah tell us that Around 70% of the trip data has a distance less than 10 kilometers and its trip duration is less than 2000 seconds which is nearly an hour. Hence, we can conclude that data contains some outliers as most of the data is compacted and there are some trips whose trip duration and distance is very huge.

<img width="890" alt="Screenshot 2023-01-18 at 11 47 58 PM" src="https://user-images.githubusercontent.com/92228157/213374331-f094eb41-f639-4e5f-bb04-e5157499f026.png">

# V. Algorithms

As mentioned above that data have several outliers, we have done some data processing to remove potential outliers before processing it with our machine learning algorithms. As both regression models require all features to be compressed into a vector, we have converted a week_day column into an index column. To extract all the features, we used VectorAssembler and converted into a PySpark DataFrame. In both regression models, we have used the trip_duration column as an output column. Around 70% of the DataFrame split into training and roughly 30% DataFrame was for left testing so, both regression models will have enough data to identify trends in the trip_duration of NYC taxi trips records.

## I. Regression Models

## II. K-Means Clustering

# VI. Results

# VII. Discussions

## Potential issues

## Future work and Closing thoughts
