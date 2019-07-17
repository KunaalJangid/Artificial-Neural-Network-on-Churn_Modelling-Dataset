#Importing the Libraries
import pandas as pd
import matplotlib as plt
import numpy as np

#Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
#Creating Dummy Variables
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#To Avoid the Dummy Variable Trap
X=X[:,1:]

#Split the Dataset into Training Set and the Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part-2 Now Let's make the Artificial Neural Network

#Importing the Keras Libraries and Packages
import keras
#Initialize Neural Network
from keras.models import Sequential
#Build the Layers of ANN
from keras.layers import Dense

#Initializing the Classifier
classifier = Sequential()

#Adding the Input Layer and the First Hidden Layer
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#Adding the Second Hidden Layer
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN i.e., applying stochastic gradient descent on the entire ANN
#The loss function used here is Logarithmic Loss Function(binary_crossentropy for 2 categories & 
#categorical_crossentropy for more than 2 categories)
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#Part 3  - Making the Predictions and evaluating the model

#Predicting the Test Set Results
y_pred = classifier.predict(X_test)

#To transform the y_pred into a variable of boolean type i.e., True or False for the confusion matrix
y_pred = (y_pred > 0.5)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)