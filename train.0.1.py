# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:19:23 2017

@author: remi
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#Load data
df=pd.read_csv("~/ML/data/EURUSDM1.csv")
df.columns=['dt','tm','o','h','l','c','v','t']

#Determine call at 5 mn
df.index=pd.to_datetime(df.dt+' '+df.tm)
df.t=df.c.shift(-1, freq='min')
df.t=df.t>df.c

#Build training dataframe
# convert [h,l,c] into 2D series. Centered on last close
def create_dataset(dataset, look_back=60):
    dataX, dataY = [], []
    for i in range(dataset.shape[0]-look_back-1):
        a=[]
        for j in range(dataset.shape[1]-2):
            a.append(dataset[i:(i+look_back), j])
        a=np.array(a)            
        a=(a-dataset[i+look_back-1, -2])/(a.max()-a.min())
        dataX.append(a)
        dataY.append(dataset[i + look_back -1, dataset.shape[1]-1])
    return np.array(dataX), np.array(dataY)
        
X,Y = create_dataset(np.matrix(df[['h','l','c','t']]['2013':'2015']))


                                                                                                                                                                                                
X=X.reshape(X.shape[:-1])


#Cutoff train from test
tCutOff=-len(df['2015'].index)
x=X[0:tCutOff]
y=Y[0:tCutOff]

xt=X[tCutOff:]
yt=Y[tCutOff:]


#Extend training set with pair inverse 
x=np.concatenate((x,-x))
y=np.concatenate((y,~y))

#Test
#y=x[:,3,30]>x[:,3,45]
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
from sklearn.metrics import roc_auc_score

#Dropout fix
import tensorflow as tf
tf.python.control_flow_ops = tf

# create and fit the LSTM network
np.random.seed(1664)
#Deep with covolution
model = Sequential()
model.add(Convolution1D(input_shape=X.shape[1:],nb_filter=32, filter_length=5, border_mode='same', activation='relu'))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
#model.add(Convolution1D(input_shape=X.shape[1:],nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(30, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
trn=model.fit(x, y, nb_epoch=10, batch_size=64)

#Prediction
yP=model.predict(xt)



#P&L
#   Treshold Filter
f=abs(yP[:,0]-.5)>.05
ytf=yt[f]


#   Profitability
c=confusion_matrix(ytf,yP[f,0]>.5)

stake=.535
c.diagonal().sum()*(1-stake) - np.flipud(c).diagonal().sum() * stake

print('Acc=',(c.diagonal().sum()/c.sum()),'AUC=',roc_auc_score(yt, yP))
#Acc= 0.557811562312 AUC= 0.53421291115

#   Equity timeline
r=(1-stake)*((yP[:,0]>.5)==yt)
r[(yP[:,0]>.5)!=yt]=-stake
r[np.invert(f)]=0

sum(r)
plt.plot(r.cumsum())
plt.plot((r/100+1).cumprod())

df.index[700000:1000000]

df.index[700000]
len(df.index)

df['2014':'2015']

plt.hist(yP)
sns.distplot(yP)

#------------------------------1378
model = Sequential()
model.add(Convolution1D(input_shape=X.shape[1:],nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Dropout(0.2))
model.add(LSTM(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, nb_epoch=100, batch_size=64)
#------------------------------

#-------------------OK: 400-600
model = Sequential()
model.add(Convolution1D(input_shape=X.shape[1:],nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
#model.add(Dropout(0.2))
#model.add(LSTM(1, activation='sigmoid'))    #Quite good
model.add(LSTM(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, nb_epoch=100, batch_size=64)
#-------------------

#Regular
model = Sequential()
model.add(Dense(5,input_dim=X.shape[-1], activation='relu'))
#model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x[:,3,:], y, nb_epoch=100, batch_size=32)

####################################
a=x[:,3,[30,45]]
b=y

model = Sequential()
model.add(Dense(5,input_dim=2, activation='relu'))
#model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(a, b, nb_epoch=100, batch_size=32)

a=np.random.random([1000,2])
b=a[:,1]>2*a[:,0]

#Regular
model = Sequential()
model.add(Dense(5,input_dim=a.shape[-1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(a, b, nb_epoch=100, batch_size=32)

aP=model.predict(a)
