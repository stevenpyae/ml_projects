# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:04:34 2021
Predicting if the cancer diagnosis is benign or malignant based on several observations/features

30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry 
  - fractal dimension ("coastline approximation" - 1)
Datasets are linearly separable using all 30 input features

Number of Instances: 569

Class Distribution: 212 Malignant, 357 Benign

Target class:

   - Malignant
   - Benign
@author: Corbi
"""

#Importing the data
# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization 
# %matplotlib inline

# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # Load the dataset 

print(cancer) 
print(cancer.keys()) # Keys within the dictionary, what columns do we have, Dictionaries do we have 

print(cancer['DESCR']) # Get the description of the Data
print(cancer['target_names']) # What are the names of the classes 
print(cancer['target']) # Show the data whether Malignant or Benign
print(cancer['feature_names']) # All the features we have
print(cancer['data'])

print(cancer['data'].shape) #Shape of the data 
#568 rows 30 columns/Features

# Create a DataFrame to represent all the data

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
# Append 30 features and Target (Dependant Variable)  tgt. 31 columns 
#Cancer data and Target data, 30 columns additional column that include all the 
print(df_cancer.head())
print(df_cancer.tail())

#Visualise the Data Pairplot, countplot and scatterplot 
# Quick glance of the entire data
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

sns.countplot(df_cancer ['target']) #how many sample is zero and how many samples is one

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer) #xaxis, yaxis, hue

#use the seaborn heat map
#figuresize to adjust the size of the figure
plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)

"""
Training the Model
"""
X = df_cancer.drop(['target'],axis=1) #dataframe, except the target value
y = df_cancer['target']

# Split the data into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5) # Fixing the Seed here

# Support Vector Machine
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False) #Parameters - Kernel, Linear Kernel, Non-linear Kernel
#Part 10, Tune your Model
svc_model.fit(X_train, y_train)

# Evaluating the Model (Maximum Margin HyperPlane)
y_pred = svc_model.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred) # True value vs Predicted 
sns.heatmap(cm, annot = True)
print(cm)
print(accuracy_score(y_test, y_pred))
# Accuracy of 0.5789

"""
Improving the Model
"""
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scl = (X_train - min_train)/range_train
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train) # Without the scaling
sns.scatterplot(x = X_train_scl['mean area'], y = X_train_scl['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scl = (X_test - min_test)/range_test

#Training and Predicting the Model again 
svc_model.fit(X_train_scl, y_train)
y_pred = svc_model.predict(X_test_scl)

cm = confusion_matrix(y_test, y_pred) # True value vs Predicted 
sns.heatmap(cm, annot = True)
print(cm)
print(accuracy_score(y_test, y_pred))
# Accuracy score of 0.96

# Plot a Classification Report
print(classification_report(y_test, y_pred))

"""
Improving the Model- Part 2
Improving the C and Gamma parameter, Grid_search
"""
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = svc_model,
                           param_grid = param_grid,
                           refit = True,
                           verbose = 4)

#List of combination of different hyper parameters
#LIst of 2 Dictionaries, one in Linear Kernel and RBF Kernel the Gamma parameter only works for RBF Kernel 
grid_search.fit(X_train_scl, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy)
#0.978 accuracy
print(best_parameters)

grid_predictions = grid_search.predict(X_test_scl) # Grid Predictions
cm = confusion_matrix(y_test, grid_predictions) # True value vs Predicted 
sns.heatmap(cm, annot = True)
print(cm)
print(accuracy_score(y_test, grid_predictions))

print(classification_report(y_test, grid_predictions))