# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:36:57 2021
Utilise the clean data that we use in the previous steps

@author: Corbi
"""

#Importing the data
# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sn # Statistical data visualization
import time

dataset = pd.read_csv('new_appdata10.csv')

#Data Preprocessing
#Split the response variable from the other independent features 

response = dataset['enrolled']
dataset = dataset.drop(columns = 'enrolled')

# Split data into trainning and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response, 
                                                    test_size = 0.20, 
                                                    random_state= 0) # Fixing the Seed here

#Removing the UserID, at the end of the model we want to associate the predictions to the users it came from
# We are going to save it away
train_identifier = X_train['user']
X_train = X_train.drop(columns = 'user')

test_identifier = X_test['user']
X_test = X_test.drop(columns = 'user')

#Feature Scaling
#returns a numpy array of multiple dimensions, it loses the column names and index
#Save it into a new dataframe and copy the column names 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

#Copy the Feature Scaled training and test set into the original after copying the column names and indexes
X_train = X_train2
X_test = X_test2

## Model Building
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
#L1 regularization model. Nature of screens, can be correlated with each other. 
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred) # number of predicted values, and number of actual real value
accuracy_score(y_test, y_pred)
# 0.768 percent
precision_score(y_test, y_pred) # tp / (tp + fp)
# 0.761 # number of true positives, divided by total true and false positives
recall_score(y_test, y_pred) # tp / (tp + fn)
# 0.770 true positives divided by true positive and false negatives 
# of all the actual real positives, how many did we predicted to be positive
f1_score(y_test, y_pred)
# 0.766 # function of the two above, good estimate to guarentee that the accuracy of our predictions are actually good

# plotting the confusion matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

#k-fold cross validation - we apply the model to different folds.
# guarentees that it works on every little subset of our data 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score (estimator= classifier, X = X_train, y = y_train, cv = 10)
print('Logistic Accuracy: %0.3f (+/- %0.3f)' %(accuracies.mean(), accuracies.std()*2))
# 0.767 (+/- 0.009)

#Formatting the Final Results
final_results = pd.concat([y_test, test_identifier], axis = 1).dropna()
final_results['predicated_results'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_reach']].reset_index(drop=True)

#### Model Tuning ####

## Grid Search (Round 1)
from sklearn.model_selection import GridSearchCV

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


## Grid Search (Round 2)

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = [0.1, 0.5, 0.9, 1, 2, 5]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
grid_search.best_score_


#### End of Model ####
