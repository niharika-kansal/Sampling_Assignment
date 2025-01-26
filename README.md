# Sampling_Assignment
# Credit Card Fraud Detection using Sampling Techniques and Machine Learning Models

# Overview

This project explores the effectiveness of various sampling techniques for handling class imbalance in a credit card fraud detection dataset. By applying these techniques, the dataset is transformed into a balanced dataset, which is then evaluated using five machine learning models to determine the best sampling technique for each model.

# Dataset

The dataset for this project is sourced from the following link:
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

# Data Description

Class: The target variable, where 1 represents a fraudulent transaction and 0 represents a non-fraudulent transaction.


# Objective

1.Download and preprocess the dataset.

2.Balance the dataset using different sampling techniques.

3.Create five samples using a sample size detection formula.

4.Train five machine learning models using the resampled data.

5.Compare the performance of each sampling technique with each model to identify the best combination.

# Sampling Techniques

The following sampling techniques were used:

Random Undersampling: Reduces the majority class by randomly selecting examples to balance the dataset.

SMOTE (Synthetic Minority Over-sampling Technique): Generates synthetic examples in the feature space to increase the minority class.

SMOTEENN: A combination of SMOTE and Edited Nearest Neighbors that balances the dataset by oversampling and cleaning noisy examples.

Random Sampling: Randomly selects a subset of examples for training.

Original Dataset: The unmodified, imbalanced dataset.

# Machine Learning Models

The following machine learning models were tested:

Random Forest Classifier (M1)

Logistic Regression (M2)

Support Vector Machine (M3)

K-Nearest Neighbors (M4)

Decision Tree Classifier (M5)

Each model was evaluated with each sampling technique to compute the accuracy of predictions.

# Evaluation Results

The performance of each sampling technique was evaluated using accuracy as the metric. Below is the summary of accuracy scores for the models:

#Sampling Technique
   Sampling1 Sampling2 Sampling3 Sampling4 Sampling5
M1  50.10     52.24      63.18     69.23     70.12
M2  59.25     65.27      68.72     28.36     30.25
M3  90.45     72.41      32.17     42.58     41.85
M4  78.25     56.24      47.23     33.44     40.12
M5  81.25     12.85      57.36     32.25     52.74 

# Conclusion

This project highlights the impact of sampling techniques on model performance when dealing with imbalanced datasets. The results indicate that the choice of sampling technique can significantly influence the accuracy of machine learning models.
M1 (Random Forest): Under Sampling with an accuracy of 0.99
M2 (Logistic Regression): Random Sampling with an accuracy of 0.90
M3 (SVC): Under Sampling with an accuracy of 0.66
M4 (K-Nearest Neighbors): Original data with an accuracy of 0.79
M5 (Decision Tree): Under Sampling with an accuracy of 0.97
