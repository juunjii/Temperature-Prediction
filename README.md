# Temperature-Prediction

## About
This project aims to develop a machine learning model capable of predicting temperature at different times based on a structured dataset. The dataset undergoes extensive preprocessing, feature engineering, and model tuning to enhance prediction accuracy. Various regression algorithms are employed to evaluate performance and ensure robustness in forecasting temperature trends.

## Data Before Cleaning
- Number of samples:  96453
- Number of features:  108
## Cleaning Data
1. Cast numerical data to float
-  Ensures consistency in the data type in the dataset to ensure that there would not be type issues in data cleaning/processing.
2. Remove duplicate samples
- Duplicate samples were removed to avoid skewing the results of model training. If the same sample appears more than once, it can give more weight in the training process, leading to biased predictions.
- Result:
  - 241 duplicate samples were removed.
3. Replace missing values
- Missing values were removed because SelectKBest does not accept NaN values in the labels. Regression algorithms also cannot handle missing values directly, which may lead to erros during training and predictions.
- Result:
  - The missing values are replaced with the median of the respective features that they lie on, because median is resistant to outliers.
4. Drop "feature_17"
- "feature_17" was dropped becasue it only contained zero values, thus it wasn't statistically significant in training the predictive model.
5. Feature Engineering
- Extracting data features from feature "Formatted Date"
  - The "Formatted Date" column was split into 7  additional features:
    - "Year"
      - Since the data spans multiple years, adding a "Year" feature would help capture long-term changes or patterns, such as steady warming or cooling over time
    - "Month"
      - Temperatures fluctuate with season (e.g., summer vs. winter), and the month is a good predictor of seasonal fluctuations. Adding a "Month" feature enables the model to learn that July is often warmer than January, for example, potentially improving its accuracy in predicting seasonal temperatures.
    - "Day"
      - Adding a "Day" feature enables for more detailed modeling within a month, as temperature patterns might change from beginning to finish.
    - "Day of Week"
      - Temperatures might vary somewhat depending on the day of the week owing to human activity. For example, weekends may have somewhat different temperature patterns if human activity influences temperature
    - "Quarter"
      - By dividing the year into quarters (Q1, Q2, Q3, Q4), the model can broadly discriminate between seasons.
    - "Hour"
      - Temperature normally cools at night and warms throughout the day. By incorporating the hour, the model may learn diurnal temperature trends, which improves forecasts, particularly for short-term forecasting.
    - "Is Weekend"
      - Weekends can cause small shifts in temperature trends due to less urban activity, particularly in densely populated places.

  After extracting all these data features, the formatted date column would be dropped.
- Extracting data features from feature "Daily Summary"
  - Adds granularity to the dataset, which is beneficial for tree-based models such as random forests, and gradient boosting (which explains the highest performance score from both models from this dataset)
  - A model may reveal that "foggy" circumstances significantly correspond with specific temperature patterns.
  - It may also determine that "breezy" circumstances affect temperature differently depending on the time of day
6. Ordinal encoding categorical columns
- Ordinal encoding categorical columns help improve model performance. Moreover, regression algorithms like SVM and Linear Regression requires numerical input so ordinal encoding categorical columns ensure compatibility with such alogrithms, so that they can be processed effectively.
- Result:
  - "Daily Summary", "Year", "Month", "Day", "Day of Week", "Quarter", "Hour" columns now have their values represented with numerics for their respective categories.
7. Removing outliers
- The lower and upper threshold is decided based on the three-sigma limits of the empirical rule, which states that almost all observed data will fall within three standard deviations of the mean on a normal distribution (bell figure).
- Outliers were removed because outliers represent noise in the data, thus removing them can help in creating a cleaner dataset that may help with training a more accurate model.
- Result:
  - 10622 outliers removed.
8. Standardize the data
-  Features are standardized to have a standard deviation of one and a mean of zero. This keeps features with bigger sizes from controlling the model and guarantees that each feature contributes equally to the distance computations in algorithms.
- Standardizing data is useful in algorithms like logistic regression, which assume normally distributed data or use distance-based calculations.
- Result:
  - Features now have a standard deviation of one and a mean of zero.
9. Feature selection
- SelectKBest with a score function of "f_regression" is used to help identify and retain the most relevant feature in a dataset for a regression problem.
- "f_regression" calculates the F-statistic for each feature, determining if there is a significant linear relationship between the feature and the target variable. This aids in finding factors that are likely to influence the goal prediction in a linear regression model.
- Result:
  - A F-score of higher than 1 is deemed to be statistically significant (decide relative o the F-score of other features), thus, 44 features were kept.


## Dimension of datagrame going into training regression model
- Number of samples: 90916
- Number of features: 44

## Regression model used:
1. Standard Linear Regression (Least Squares)
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor

For all regression algorithms used, the dataset is prepared for training by extracting features and labels, then splitting them into training and testing sets (20% test, 80% training). This specific distribution is chosen because it is widely used likely due to its ability to enable the model to discover the links and underlying patterns in the data. A training set of 80% offers enough data for the model to learn efficiently.

A compromise between keeping enough data to test the model and having enough data to train it is achieved by the 80/20 split. A small training set could prevent the model from learning well, which would result in subpar performance. On the other hand, a testing set that is too small might not offer a reliable evaluation of the model's performance.

Training the regression model begins with hyperparameter tuning by cross validation.

For Standard Linear Regression (Least Squares), there was no need to perform cross validation since there are no hyperparameters to tune.

For Ridge, Lasso Regrssion, and Gradient Boosting Regressor Models, "GridSearchCV" was used to find the best hyperparameters by passing in a dictionary defining the parameters and range of values you want to tune. For instance, for Ridge Regression, alpha values of 0.1, 1, 10, 100 were passed into the function to determine which alpha value produces the highest performance score. 5-fold cross-validation is chosen when using "GridSearchCV" since it tunes multiple hyperparameters, and having fewer folds can make model training faster. The same can be said when training a Gradient Boosting Regressor model, as a set of three hyperparameters with 3 different possible values are being passed into the "GridSearchCV" function - {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}.

For Random Forest Regressor, I used the "cross_val_score" function to perform cross valdiation on hyperparameter tuning. A  10-fold cross-validation is used as it offers a slightly lower variance in model performance estimates, when the data is split into 10 different training-validation sets.

Hyperparameter tuning by cross validation is important because it decreseas result variance and helps mitigate overfitting. By identifying hyperparameter settings that work well over several folds, cross-validation lessens the possibility of choosing parameters that just work well on a particular training set.


### Performance Metrics (mention why u chose based on advantages)
MAE (Mean Absolute Error)
- calculates the average size of mistakes in a group of predictions without regard to their direction.
- A useful indicator for understanding how distant the forecasts are, on average, from the actual data.
- The smaller the MAE, the better the model.

RMSE (Root Mean Square Error)
- easier to interpret because it gives the error in the same unit as the target variable
- The smaller the MSE, the better the model.

R^2
- measures the proportion of the variance in the target variable that is predictable from the features.
- demonstrates how effectively the model accounts for the data's variability.
- The higher the MSE, the better the model.


I chose to not use MSE (Mean Squared Error) as a performance metric because the data is not scaled back to its original scale, so the temperature predicted would be degree celcius squared instead of degree celcius, thus harder to interpret.

Instead, I use RMSE (Root Mean Squared Error), because it scales back the data to the original scale, so the temperature predicted would be degree celcius. Hence, making it easier to interpret and more intuitive for assessing error size. Moreover, it also has a greater sensitivity to larger errors, as it penalizes larger errors more than smaller ones, making it useful when large deviations from actual values are unwanted.

I chose to use MAE (Mean Absolute Error) because the result is expressed as the same scale as the original data (degree celcius in this case). It is also a good metric to use because erros are penalized equally so it is more robust to outliers than MSE. However, this shouldn't matter because I removed outliers during data cleaning. Furthermore, MAE treats all errors equally by calculating the absolute difference, avoiding excessively penalizing greater errors, making it a good metric to include alongside with RMSE, which peanalizes large errors.

I use R^2 because it is a relative metric that scales with the dataset, and can be used to compare against other models that I trained in this dataset, so I can assess model's performance under the same scale.


### Summary of Hyperparameters Chosen and performance scores
- Linear Regression:
  - No hyperparameter
  - MAE=4.33, RMSE=5.48, R^2=0.68
- Ridge Regression:
  - Best hyperparameter: {'alpha': 100}
  - MAE=4.33, RMSE=5.48, R^2=0.68
- Lasso Regression:
  - Best hyperparameter: {'alpha': 0.01}
  - MAE=4.33, RMSE=5.48, R^2=0.68
- Random Forest:
  - Best hyperparameter: {'num_tree': 50}
  - MAE=2.19, RMSE=2.91, R^2=0.91
- Gradient Boosting:
  - Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
  - MAE:  MAE=2.38, RMSE=9.18, R^2=0.90

With 60% of the variance explained, the linear regression model exhibits a decent level of performance. The RMSE and MAE numbers, however, indicate that there is potential for improvement.

In this instance, ridge regression with alpha = 100 did not outperform the linear regression model. There are several possible explanations for this, including the dataset's relative simplicity or the lack of significant multicollinearity among features that regularization might enhance.

With a very low regularization strength (alpha = 0.01), Lasso regression also yielded results that were comparable to those of Ridge and Linear regression. There may not be enough regularization for the model to perform noticeably better.

Random Forest performs much better than Linear, Ridge, and Lasso regression. With an R² of 0.91, the model is capturing most of the variability in the target variable and making significantly more accurate predictions

Gradient Boosting is also a very strong model with an R² of 0.90, indicating it performs very well in explaining the variance in the target variable. However, it is slightly less accurate than Random Forest (as evidenced by the MAE and R² scores).

Compared to Lasso, Ridge, and Linear regression, Random Forest performs significantly better. The model is producing noticeably more accurate predictions and capturing the majority of the variability in the target variable, as indicated by its R2 of 0.91.

With an R2 of 0.90, gradient boosting is likewise a very powerful model that does a great job of explaining the variance in the target variable. The MAE and R2 scores, however, show that it is marginally less accurate than Random Forest.


## Lessons Learned

I learned how to write scalable code that is more effienct and able to leveraged Spark's distributed computatating ability to process large datasets.
Previously (in hw3), to identify and remove outliers, I was looping over each column individually, collecting rows locally, and adding the row indices to a set. This combination of operations proved to be slow as PySpark is not designed for row-wise operations, but vectorized operations on entire columns or bathces of rows in parallel across a cluster (group of machines or servers that work together to execute a task).

Hence, when working with larger datasets, I implemented these changes in order to optimize my code and improve run time:

1. Took advantage of functional programming techniques, by using functions like reduce, which allowed me to combine multiple conditions into a single expression.
2. Used single-pass filtering with a combined condition, which managed to minimzie the number of operations Spark needed to perform, saving processing time and memory.
3. Broadcasted variables (outlier_indices) allowed me to distribute a small set of data across nodes, speeding up operations that involve lookups or comparisions across nodes. This is due to the fact that distributed processing using Spark can parallelize jobs to accommodate datasets larger than memory, but only if we optimize our code to prevent complex operations and excessive data transmission.
4. Filtering data in Spark and gathering the bare minimum of required results (row indicators of outliers) reduced the amount of data transferred to the driver. By doing this, network expenses are kept low and the driver node is not overloaded with data.

After implementing said changes, I was able to remove my outliers in 1 minute, where it took me 2 hours previously.

## For the future

With the lessons learned from above, this is how I will leveraged them in the future:

- Scalable Data Processing

I'll be able to manage bigger datasets now that I understand how to create effective, distributed code. I can process and analyze large datasets fast and effectively without overtaxing the system by using techniques like vectorized operations, which are essential for performance and resource management. With these skills learned, I would be using PySpark to process large datasets instead of Pandas because it leverages distributed computing across a cluster of machines, allowing for large amounts of data to be processed effectively.

- Resource Management in Cloud Environments

I will be able to use these resource management strategies in cloud applications because Spark's distributed architecture is compatible with a lot of cloud-based systems. I can create scalable, high-performance apps in any industry that deals with massive amounts of data because of my experience with optimizing code for distributed systems.


- Optimized Data Transfer

I will be able to perform better on distributed computing platforms if I know how to minimize driver load and broadcast variables to lessen data transportation. I'll be able to operate in cloud environments and with frameworks like Hadoop, where reducing data transit can save money and boost productivity.

- Functional Programming Techniques

Using functional programming approaches, such as reduce, will be helpful not only in Spark but in any situation where operations need to be parallelized or streamlined. I'll use these abilities to develop intricate ETL procedures and strong data pipelines, which are crucial in data engineering and analytics positions where productivity is crucial.


## Conclusion

### Best model to use:
Random Forest, because it has the lowest MAE and RMSE, and the highest R^2 square.

Best hyperparameter: {'num_tree': 50}

MAE=2.16, MSE=8.26, RMSE=2.87, R^2=0.91

Overall, the random forest model scored the best because I performed feature engineering on the "Daily Summary" and "Formatted Date" feature, thus adding granularity to the dataset. By catching subtleties that wider features would overlook, more granular features might more precisely depict underlying patterns. For instance, a model might learn patterns linked to particular times or dates by decomposing a timestamp into smaller components, such as hour, weekday, and month. This can increase the forecast accuracy of the model. (explained more in depth in "5. Feature Engineering" in the "Cleaning Data" section of the report.)
