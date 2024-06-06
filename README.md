# Objective
The objective of this machine learning project is to build and evaluate predictive models to classify the target variable y in the provided dataset. The dataset is related to bank marketing, and the target variable indicates whether a client has subscribed to a term deposit (yes or no) after a campaign.

# Steps and Process

## Data Loading and Exploration
The dataset was loaded and explored to understand its structure and contents.
Transformation are completed to encode categorical features

## Data Encoding
- Categorical variables were encoded to numerical values:
  - default, housing, loan, and y were encoded as yes -> 1, no -> 0, and unknown -> -1.
- Months were encoded from 1 to 12.
- Days of the week were encoded from 1 to 7.
- job, marital, and education were encoded with numerical sequences.
- poutcome and contact were encoded with specified mappings.

## Train-Test Split
The dataset was split into training and testing sets with an 80-20 split.

## Baseline Performance
The baseline performance was established using the majority class classifier, which achieved an accuracy of approximately 88.65%.

## Model Training and Evaluation
A Logistic Regression model was trained and evaluated, achieving an accuracy of approximately 90.93%.
Further scaling of features was performed to improve convergence.
Additional models (K-Nearest Neighbors, Decision Tree, and Support Vector Machine) were selected for comparison.

## Hyperparameter Tuning using GridSearchCV
Hyperparameter tuning was performed using GridSearchCV for each model to find the best set of parameters. The time taken for each process was also calculated.

### Hyperparameters for Each Model:

#### Logistic Regression

    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']

#### K-Nearest Neighbors

    'n_neighbors': [3, 5, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']

#### Decision Tree

    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 4, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']

#### Support Vector Machine

    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']


## Model Comparison

Models were compared based on train time, train accuracy, and test accuracy.
The best performing model in terms of test accuracy was identified.

# Results
Each model was evaluated and tuned using the steps described above. Results included the best hyperparameters and the time taken for each model to fit and tune.
By following these steps, all models performed with a training accurary around 92% and test accurary close to 91%.
Time invested in tuning the hyperparameters varied significantly, as well as the training time.
As the accuracy results are very similar across models, the reccommendation would be based on training times. While the time invested in the hyperparameter tuning was high (638 seconds) with the KNN model, the train time once identified the best parameters was optimal (0,008 seconds).

Best Params for KNN: {'algorithm': 'auto', 'n_neighbors': 9, 'weights': 'uniform'}
Time taken for KNN Grid Search: 638.76 seconds
Train time: 0.008064985275268555
Train accuracy: 0.9192412746585736
Test accuracy: 0.9031318281136198

