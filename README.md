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

# I. Introduction

New York City (NYC) is one of the biggest cities in the world, with a population of around 18,000,000 people. Due to its large population, traffic congestion is one of the most critical problems in New York. In addition, taxi cabs are a popular choice of transport when traveling within this city. That is why in this project, we chose to predict the trip duration for a taxi in New York using the “NYC taxis trip duration dataset” which can be found <a href="https://github.com/adam-25/NYC-taxi-Trip-duration-BigData-Analysis/blob/main/Data/NYC%20Taxi%20Duration.csv" target="_blank">here</a> and so people can get an idea of how long their taxi trip will take and plan their trips beforehand. 

The current dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo coordinates, number of passengers, and several other variables. The dataset contains more than 510,000 rows. Since the dataset is mostly from a Kaggle challenge (where people compete to use the provided data to create the best visualizations, machine learning models and reports), there are a lot of machine learning models that were built using this dataset. However, we found out that the accuracy of these models is not very high, which means that there is a possibility of developing models that can achieve higher accuracy and reliability. Moreover, the current machine learning models were built using mostly Tensorflow and other CNN techniques, while we manipulated the data and create a model using SkLearn and PySpark.

In our solution, we have created several visualizations that show a relationship between trip duration and other variables, also we demonstrated how the mean passenger count, total trips, distance and duration vary on different weekdays. Also, we have created 3 machine learning models to either gain insights into the structure of the data or try to predict the duration of a trip. That includes regression models like linear regression, gradient boosted trees, and K-means clustering.

# II. Data Preprocessing

While preparing the data for the machine learning models, there were several operations performed on the dataset. We have changed the type of each column from string to either integer, double, string or timestamp according to its values. Also, we have calculated the distance between pickup latitude, longitude and dropoff latitude and longitude for all the trips. As we believe that taxi trips in New York City might not take more than 2 hours at any time we filtered only trips whose trip duration is between 3 minutes to 2 hours. In addition to that we believe that taxi rides in New York city may always be less than 60 kilometers as no taxi accepts rides as the destination is more than 60 kilometers far away. There are also several preprocessing has been done on the data before using it for the machine learning models.

<img width="902" alt="Screenshot 2023-01-18 at 11 27 15 PM" src="https://user-images.githubusercontent.com/92228157/213371082-3b847139-5b3e-4e9d-9acc-9b80532264a1.png">

<img width="981" alt="Screenshot 2023-01-18 at 11 38 41 PM" src="https://user-images.githubusercontent.com/92228157/213372876-aa1a338e-65b2-4e54-bc1c-f0bf05da54d7.png">

# III. Data Visualization

To get good insights from the existing dataset, we have performed several data visualization. From the figure below, we can see that the total number of trips is higher on Friday and Saturday with the Average Passenger count higher on Sunday as well because people are more likely to go out with the family at the start of the weekend. While another grpah helps us to match our initial expectation that trip duration is lower on weekends as there would be less traffic because most of people are not working.

<img width="780" alt="Screenshot 2023-01-18 at 11 45 25 PM" src="https://user-images.githubusercontent.com/92228157/213373986-946fc068-cdaa-4a6a-a953-5f0068647283.png">

<img width="732" alt="Screenshot 2023-01-18 at 11 45 42 PM" src="https://user-images.githubusercontent.com/92228157/213374030-a9cf4162-f2ce-4afe-aafd-080e293bdf1c.png">

While predicting and getting detailed insights of the data regarding any duration of the trip the most important variable to consider is a distance and also most people rely on the distance of two places to predict their duration of the trip. We have plot the relationship between distance and trip duration. Also, grpah tell us that Around 70% of the trip data has a distance less than 10 kilometers and its trip duration is less than 2000 seconds which is nearly an hour. Hence, we can conclude that data contains some outliers as most of the data is compacted and there are some trips whose trip duration and distance is very huge.

<img width="890" alt="Screenshot 2023-01-18 at 11 47 58 PM" src="https://user-images.githubusercontent.com/92228157/213374331-f094eb41-f639-4e5f-bb04-e5157499f026.png">

# IV. Algorithms

As mentioned above that data have several outliers, we have done some data processing to remove potential outliers before processing it with our machine learning algorithms. As both regression models require all features to be compressed into a vector, we have converted a week_day column into an index column. To extract all the features, we used VectorAssembler and converted into a PySpark DataFrame. In both regression models, we have used the trip_duration column as an output column. Around 70% of the DataFrame split into training and roughly 30% DataFrame was for left testing so, both regression models will have enough data to identify trends in the trip_duration of NYC taxi trips records.

### I. Regression Models

The first regression algorithm that we used is linear regression from PySpark's machine learning library. It is supervised learning based on the representation of the equation of the straight line which is `y = mx + c`. We used VectorAssembler to get the feature results. We have used ParamGridBuilder and CrossValidator to experiment a model with different hyperparameters like regParam, elasticNetParam, maxIter, and fitIntercept.

The second regression algorith that we used is gradient boosted trees regression from PySpark's machine learning library. It is also a supervised learning that calculates the difference between the current prediction and the known correct target value. This difference is called residual. After that Gradient boosting Regression trains a weak model that maps features to that residual. This residual predicted by a weak model is added to the existing model input and thus this process nudges the model towards the correct target. Repeating this step again and again improves the overall model prediction. Similar to linear regression, we used VectorAssembler to get the feature results. We have used ParamGridBuilder and CrossValidator to experiment a model with different hyperparameters like maxDepth, maxIter, and maxBins.

### II. K-Means Clustering

The second algorithm we used is KMeans clustering from Python's SKLearn library. The K-means clustering is an unsupervised machine learning technique that groups similar data points together based on underlying similarities and patterns. Aim of the KMeans clustering was to find out if data points within the same clusters have similar trip durations or not. If any correlation was found, it becomes easier to find out which properties of the dataset are strongly influencing the trip duration. We used "Elbow method to calculate the optimum number of clusters. We calculated Within Cluster Sum of Squares (WCSS) which means the sum of squared distance between each point and the centroid of the cluster.

<img width="1019" alt="Screenshot 2023-01-19 at 10 17 31 PM" src="https://user-images.githubusercontent.com/92228157/213622672-35031bb7-93c4-43c6-bd10-3ec9518ba974.png">

Hence, we concluded the optimal number of cluster k = 4. With the optimal number of cluster, we have done a cluster dianoses to check which cluster has the most of the datapoints and to check its magnitude. Result of our diagnoses is below.

<img width="1092" alt="Screenshot 2023-01-19 at 10 20 41 PM" src="https://user-images.githubusercontent.com/92228157/213623035-e386e671-2f77-4152-bde1-e5499d7bac8f.png">

# V. Results

### I. Linear Regression

With this model, we have got some best parameters for the model and achieved an accuracy of the 50.23% with the root mean square error(RMSE) of 500.76. In the below picture, we can observe some of the best parameters for the linear regression model along with the coefficients and intercept of the linear line.

<img width="1170" alt="Screenshot 2023-01-19 at 10 37 44 PM" src="https://user-images.githubusercontent.com/92228157/213625033-a1aba730-891d-43f3-a161-f450b762883e.png">

### II. Gradient Boosted Trees Regression

With the best hyperparameters for the gradient boosted trees regression model, we achieved an accuracy of roughly 75% with the root mean square error(RMSE) of 362.1. In theory, gradient boosted trees regression works better than linear regression and after checking this values, we reached to the conclusion that gradient boosted trees model is better than linear regression to predict the taxi trip duration in New York City. However, the best hyperparameter for this model is shown below.

<img width="883" alt="Screenshot 2023-01-19 at 10 42 14 PM" src="https://user-images.githubusercontent.com/92228157/213625609-77f676a2-13f1-4e36-b257-c470df74a3ca.png">

### III. K-Means Clustering

For clustering, it was difficult to see the differences in clusters. Although we performed many iterations while changing different parameters and as mentioned above we found the ideal number of clusters were k = 4. Also, for a business person and a data scientist, it is important to identify the key characteristics for each cluster in order to know what each cluster stands for. Hence, we have generated a box plot that shows each feature distribution per cluster.

<img width="1095" alt="Screenshot 2023-01-19 at 10 47 01 PM" src="https://user-images.githubusercontent.com/92228157/213626233-0055d8cb-0d5f-481e-81de-35a2fc65e1d2.png">
<img width="1095" alt="Screenshot 2023-01-19 at 10 48 34 PM" src="https://user-images.githubusercontent.com/92228157/213626427-f2018824-22a5-42b6-b2ff-cef75e8335f5.png">
<img width="1074" alt="Screenshot 2023-01-19 at 10 48 54 PM" src="https://user-images.githubusercontent.com/92228157/213626465-8a868bbd-9797-46fa-91d4-41bb697a13c8.png">

We later pre-processed the data and generated a radar plot, which is another best way to summarize all relevant information in one plot. From radar plot, we concluded that a datapoint to be in cluster 1 and 3 highly depends on the distance, dropoff_longtitude, pickup_logtitude and passenger_count respectively. While data points to be in cluster 2 highly depended on the month and quarter of the year.

<img width="712" alt="Screenshot 2023-01-19 at 10 53 43 PM" src="https://user-images.githubusercontent.com/92228157/213627087-3733bc39-0817-48d5-a5a8-9afdb77bbfbc.png">

With scikit-learn, we were able to better visualize our results and perceive differences in clusters but 75% accuracy result from our Gradient Boosted trees regression model, which isn't very high, but enough to be interesting and for future work.

# VI. Discussions

Numerically, with 75% accuracy, our Gradient Boosted trees regression model doesn't seem tremendously "accurate" or "precise". However, with the limited resources and as it takes huge time for the machine learning model to get train, we think achieving 75% accuracy is better than other solutions with the lower accuracy and higher RMSE.

## Potential issues

- Limited Hardware: To use this huge data set and build machine learning models, it requires to have really good hardwares with good unified memory, better SSD storage and most importantly a great GPU. We have limited hardware resources, hence we were not able to experiment our machine learning model with several different hyper tuning parameters and as it takes lot of time to train machine learning model with more hyperparameters we restricted our machine learning model with the highest accracy of 75%.

## Future work and Closing thoughts

- As for utility, we believe our solution, with just a bit more tuning, can actually be useful and create value. We believe this would be of great interest to those, who like to predict NYC taxi trip duration during peak hours.

- For the future work, we planned to use cloud services like AWS or GCP to host our data and create machine learning models so, we do not have to restrict ourselves and our machine learning models. Also, in the future we plan to get real time data from some of the open source API and build a pipeline so, our model improves with the time and more important data patterns.
