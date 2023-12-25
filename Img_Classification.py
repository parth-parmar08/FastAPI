import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,BatchNormalization,MaxPool2D
from keras.models import Sequential
import tensorflow as tf
import keras

#import os
#for dirname, _, filenames in os.walk('American Sign Language Digits Dataset'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))



sign0 = pd.read_csv("American Sign Language Digits Dataset/0/Output Images - Sign 0.csv")
sign1 = pd.read_csv("American Sign Language Digits Dataset/1/Output Images - Sign 1.csv")
sign2 = pd.read_csv("American Sign Language Digits Dataset/2/Output Images - Sign 2.csv")
sign3 = pd.read_csv("American Sign Language Digits Dataset/3/Output Images - Sign 3.csv")
sign4 = pd.read_csv("American Sign Language Digits Dataset/4/Output Images - Sign 4.csv")
sign5 = pd.read_csv("American Sign Language Digits Dataset/5/Output Images - Sign 5.csv")
sign6 = pd.read_csv("American Sign Language Digits Dataset/6/Output Images - Sign 6.csv")
sign7 = pd.read_csv("American Sign Language Digits Dataset/7/Output Images - Sign 7.csv")
sign8 = pd.read_csv("American Sign Language Digits Dataset/8/Output Images - Sign 8.csv")
sign9 = pd.read_csv("American Sign Language Digits Dataset/9/Output Images - Sign 9.csv")

df = pd.concat([sign0, sign1, sign2, sign3, sign4, sign5, sign6, sign7, sign8, sign9])

features = df.iloc[:,1:-1]
labels = df.iloc[:,-1]

features_np = np.array(features)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
x = sc_X.fit_transform(features_np)
y = labels

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from sklearn import metrics

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
cv = KFold(n_splits=10, random_state=1, shuffle=True)

from sklearn.neighbors import KNeighborsClassifier

modelKNN = KNeighborsClassifier(n_neighbors=10)
modelKNN.fit(x_train, y_train)

y_predKNN = modelKNN.predict(x_test)

scores = cross_val_score(modelKNN, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
accuracy_KNN = (mean(scores)*100)
print('Accuracy (KNN) : %.2f' % accuracy_KNN, '%')

filename = 'KNN_model.sav'
pickle.dump(modelKNN, open(filename, 'wb'))