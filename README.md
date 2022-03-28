# Credit_Risk_Analysis

# Overview
This project required examining credit card data for peer-to-peer lending service company LendingClub to determine credit risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, different techniques were utilized to train and evaluate models with unbalanced classes. Imbalanced-learn and scikit-learn libraries were used to build and evaluate models using resampling. To accurately predict the credit risk, the following algorithm was used: 

For over sample
- RandomOverSampler
- SMOTE

For under-sample 
- ClusterSentriods

For over-and-under
- SMOTEENN

Lastly, two machine learning models were used to reduce bias, BalancedRandomForestClassifier, and EasyEnsembleClassifier, to predict credit risk.

# Data
- LoanStats_2019Q1.csv
- imbalanced-learn documentation
- scikit-learn documentation

# Resources
- Jupyter Notebook
- Python v3.x
  - Dependencies
    - Numpy
    - Pandas
    - Pathlib
    - Collections
    - SKLearn
    - ImbalancedLearn
# Results
Credit card data from LoanStats_2019Q1.csv was cleaned before machine learning techniques were applied. Null columns and rows were removed, interest rates were converted to numerical values, and the target column (y-axis) was converted to low_risk and high_risk based on their values.

After cleaning the data, it is divided into training and test categories, resulting in four sets of data:
X_train
x_test
y_train
y_test

A random_state of 1 was used in all models to ensure reproducible output.

The balance between low risk and high risk is imbalanced, but this was expected because credit risk is an inherently unbalanced classification problem, as good loans easily outweigh risky loans.



# Oversampling algorithm
## Naive random oversampling

![Screen Shot 2022-03-27 at 11 34 37 PM](https://user-images.githubusercontent.com/94031446/160322132-6ad8fff7-1449-4db9-8812-46e8437a0031.png)

A balanced accuracy score of 0.644 means that 35.6% of the squares are incorrect and 64.4% are correct.

A mean accuracy score of 0.99 means that this model determined the number of positive class predictors that belonged to the positive class 99% of the time.


A mean recall score of 0.67 means that this model determined the number of positive class predictions made from all positive examples 67% of the time.

SMOTE Oversampling
In the Synthetic Minority Oversampling Technique (SMOTE) oversampling model, the minority (high risk) class is repeated before the model is fitted. It can balance the class distribution but does not provide any additional information to the model. SMOTE selects adjacent data points in the feature space, draws a line between the points in the feature space, and draws a new sample at a point along that line. Realistic high_risk data is created that is relatively close to existing high-risk data.

After the data is balanced and trained, SMOTE oversampling yields the following scores:

Balanced Accuracy: 0.648

![Screen Shot 2022-03-27 at 11 36 14 PM](https://user-images.githubusercontent.com/94031446/160322277-d9a4a969-777b-449c-ac8a-569712848895.png)

The balanced accuracy score for this model means that 64.8% of the squares are correct and 35.2% are incorrect.

A mean accuracy score of 0.99 means that this model predicted positive class predictions 99% of the time.
A mean recall score of 0.64 means that 64% of class predictions made from all positive examples in the dataset were correct and 36% were incorrect.
Comparing the performance of the naive random oversampling and SMOTE oversampling models, they seemed to perform about the same.

# Undersampling algorithm

## ClusterCentroids
The ClusterCentroid algorithm provides an efficient way to represent data clusters with a small number of samples. A cluster is a group of data points grouped because of some similarity. This algorithm does this by performing K-means clustering on the majority class, Low_risk, and then creating new data points that are the average of the coordinates of the generated cluster.
After the data was balanced and trained, subsampling the ClusterCentroids yielded the following scores:

Balanced Accuracy: 0.644

![Screen Shot 2022-03-27 at 11 38 19 PM](https://user-images.githubusercontent.com/94031446/160322430-da5e89fe-6c3c-4433-9301-f648852317c6.png)


The balanced accuracy score for this model was 0.644, meaning that 35.6% of the squares are incorrect and 64.4% are correct.
A mean accuracy score of 0.99 means that the ClusterCentroid algorithm predicted positive class predictions 99% of the time on this dataset.
A mean recall score of 0.67 means that 67% of class predictions made from all positive examples in the dataset were correct, while 33% were incorrect.

# Composite sample
## SMOTENN
The SMOTEENN algorithm is a combination of the SMOTE and Edit Nearest Neighbor (ENN) algorithms. Simply put, SMOTEENN randomly observes the minority class (high risk) and the majority class (low risk)
After the data is balanced and trained, the SMOTEEN algorithm obtains the following scores:
Balanced Accuracy: 0.644

![Screen Shot 2022-03-28 at 1 36 33 AM](https://user-images.githubusercontent.com/94031446/160332975-72eedea9-e4aa-47ef-b8d0-8224479f0daf.png)


SMOTEENN had a balanced accuracy score of 0.644, meaning that 64.4% of the class predictions were correct and 35.6% were incorrect.

An average accuracy score of 0.99 is given by the SMOTEENN algorithm for positive class prediction 99% of the time in the dataset.

The mean dataset was found to have a mean recall score of 0.67, with 67% of all positive examples being correct, while 33% were incorrect.

# Ensemble Learners
## Balanced Random Forest Classifier
The Balanced Random Forest Classifier is an ensemble method where each tree in the ensemble is built from a sample drawn with replacement (bootstrap sample) from the training set. Instead of using all the features, a random subset of features is selected, which further randomizes the tree. As a result, the bias of the forest increases slightly, but since the less correlated trees are averaged, its variance decreases, which results in an overall better model.

After the data is balanced and trained, the following numbers are obtained using the balanced random forest algorithm:

Balanced Accuracy: 0.788

![Screen Shot 2022-03-28 at 1 38 16 AM](https://user-images.githubusercontent.com/94031446/160333123-449452a6-0a4e-42b9-907e-b92e36ce0115.png)


The balanced accuracy score of the Yo algorithm was 0.788g, meaning that approximately 79% of the class predictions were correct and 21% incorrect.

A mean accuracy score of 0.99 is assigned to the balanced random forest. Let the algorithm predict the positive classes in the dataset 99% of the time.

A mean recall score of 0.91 means that 91% of all positive examples in this dataset should be correct, while 9% were incorrect.

# Easy Ensemble AdaBoost Classifier

The Easy Ensemble AdaBoost Classifier combines multiple weak or low accuracy models to create a strong, accurate model. This algorithm uses one-level decision trees as weak learners that are added to the ensemble sequentially. This is an iterative process, so each subsequent model attempts to correct predictions made by the previous model in the sequence.

The data was balanced and trained using the old Easy Ensemble AdaBoost classifier algorithm to obtain the following scores:

Balanced Accuracy: 0.672

![Screen Shot 2022-03-28 at 1 38 48 AM](https://user-images.githubusercontent.com/94031446/160333179-a76955f9-509d-45da-9cf1-9d03f5ee2a7b.png)


The Easy Ensemble AdaBoost classifier achieved an accuracy score of 0.925, which means it got predictions right 92.5% of the time and 7.5% of the time wrong.

The algorithm has an accuracy score of 0.99, which means that the dataset will be predicted as positive class predictors 99% of the time.

A mean recall score of 0.94 was found to be 94% correct for all positive examples in the dataset.

# Summary
The oversampling, undersampling, and combination sampling algorithms were relatively the same. Balanced Random Forest Classifier had a higher balanced accuracy score than the previous algorithms tested, but it was not good enough for predicting credit risk.
Out of the six supervised machine learning algorithms tested, Easy Ensemble AdaBoost CLassifier performed the best overall. It had a balanced accuracy score, along with high precision and recall scores. It also had a high specificity score, which means this algorithm correctly determined actual negatives 91% of the time, and a high F1 score. This means the harmonic mean of precision and recall was 0.97 out of 1.0.
