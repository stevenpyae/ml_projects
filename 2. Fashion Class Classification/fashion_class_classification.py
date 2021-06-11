# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:27:16 2021
Fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. 
Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes.

The 10 classes are as follows:
0 => T-shirt/top 1 => Trouser 2 => Pullover 3 => Dress 4 => Coat 5 => 
Sandal 6 => Shirt 7 => Sneaker 8 => Bag 9 => Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. 
Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel,
 with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.
@author: Corbi
"""

'''Step 1: Importing'''
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

'''Step 2: Importing the Data'''
fashion_train_df = pd.read_csv("fashion-mnist_train.csv", sep = ',')
fashion_test_df = pd.read_csv("fashion-mnist_test.csv", sep = ',')

'''Step 3: Data Visualisation'''

#Exploring the Training dataset
fashion_train_df.head()
fashion_train_df.tail()

#Explore the Testing dataset
fashion_test_df.head()
fashion_test_df.tail()

#Shape of the Dataset
fashion_train_df.shape
fashion_test_df.shape

training = np.array(fashion_train_df, dtype = 'float32')
testing = np.array(fashion_test_df, dtype = 'float32')

#Random Visualising of the inputs in the Training Set
import random
i = random.randint(1,60000)
#Visualising the dataset reshape the data into 28x28
plt.imshow(training[i, 1:].reshape(28,28)) # 1: because row 0 is Label, hence, take from row 1 onwards 
# Display the Labels for it. To understand what is the data represent
label = training[i,0]

#The 10 classes are as follows: 0 => T-shirt/top 1 => Trouser 2 => Pullover 3 => Dress 4 => Coat 5 => 
#Sandal 6 => Shirt 7 => Sneaker 8 => Bag 9 => Ankle boot

# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 15
L_grid = 15

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(training) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4) # hide between the images


'''Step 4: Covolutional Neural Network Training Pre-process data '''
X_train = training[:, 1:]/255 # Normalisatio
y_train = training[:, 0]

X_test = testing[:, 1:]/255 # Normalisatio
y_test = testing[:, 0]

#Validation Dataset that is to be used during training 
# This is to avoid Overfitting 

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 12345)

# Put them in a form for inputing into CNN layer 28x28x1
# Reshaping our data. 
# Feed our images into a convolutional Network
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

X_train.shape #Greyscale image 28x28
X_test.shape #Greyscale image 28x28
X_validate.shape #Greyscale image 28x28


'''Step 4: Covolutional Neural Network Training '''
# Import Keras
import keras
from tensorflow.keras.models import Sequential # Sequential Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard 


cnn_model = tf.keras.models.Sequential()

# Applying Convolution Layer
cnn_model.add(Conv2D(32, 3 , 3 , input_shape = (28,28,1), activation='relu')) 
# 32 Kernels, size 3x3
# Size of our input shape 
# And activation layer 

#Applying Max Poolig 
cnn_model.add(MaxPooling2D(pool_size = (2,2)))
# Pool size is 2x2

#Flatten our model
cnn_model.add(Flatten()) #flatten into 1d array to feed it to our Neural network

#Applying a Hidden Layer
cnn_model.add(Dense(32, activation = 'relu'))
#Specify our dimensions output_dim = 32

#Applying the Output Layer, 10 classes, and Sigmoid Function
cnn_model.add(Dense(10, activation = 'sigmoid'))

#Using our Adam Optimizers and Compile 
opt = Adam(lr = 0.001)
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer = opt , metrics =['accuracy'])
# loss is based for 10 classes - sparse categorical 
# Matrics is accuracy

#Spcify how many epochs
epochs = 50 #How many times are we presenting the dataset and UPDATING THE WEIGHTS Iterations 8
cnn_model.fit(X_train, y_train, batch_size = 512, 
              epochs = epochs, 
              verbose = 1,  
              validation_data = (X_validate, y_validate))
# Batch Size can be Any
# Epochs 50
# How many information you need while you are training
'''Step 5: Evaluating the Model '''
#Evaluating the Model
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
# Test accuracy of 0.868 and Loss of 0.359

# get the predictions for the test data
predicted_classes = cnn_model.predict_classes(X_test)

# Print the Grid that contains the True Label and Predicted Label
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):   #Zero till the 25 images
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)


#Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)
# Sum the diagonal element to get the total true correct values


# Classification Report 
from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names = target_names))

'''              precision    recall  f1-score   support

     Class 0       0.79      0.86      0.82      1000
     Class 1       0.97      0.97      0.97      1000
     Class 2       0.79      0.78      0.79      1000
     Class 3       0.88      0.89      0.89      1000
     Class 4       0.75      0.84      0.79      1000
     Class 5       0.97      0.93      0.95      1000
     Class 6       0.69      0.54      0.61      1000
     Class 7       0.91      0.94      0.92      1000
     Class 8       0.97      0.97      0.97      1000
     Class 9       0.94      0.95      0.95      1000

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000'''