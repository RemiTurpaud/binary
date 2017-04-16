# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:19:23 2017

@author: remi

Train from tick data

"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy_groupies as npg

#Load data
df=pd.read_csv("~/ML/FX/data/EUR_USD.csv",usecols=['RateDateTime','RateBid','RateAsk'],dtype={'RateBid': np.float32,'RateAsk': np.float32},index_col=0,parse_dates=True)
df['c']=(df.RateBid+df.RateAsk)/2
df.drop(['RateBid','RateAsk'],axis=1,inplace=True)

#Resample to second
df=df.c.resample('1s').ohlc().close
#df=pd.DataFrame(df.c.resample('1s').ohlc().close)

#Build training dataframe
# convert [h,l,c] into 2D series. Centered on last close
def create_dataset(dataset, look_back=60,look_ahead=3,sample_per=60,sample=.05):
    idx, dataX, dataY = [], [], []

    #Static re-sampling index
    ix=np.floor(np.linspace(0,look_back,look_back*sample_per+1)[0:-1]).astype('int')

    #For each bar, extract re-sampled history and target
    for i in range(dataset.shape[0]-(look_back+look_ahead)*sample_per-1):

        #Print Progress
        if i%100000 == 0:
            print(i,'rows processed')
        
        if np.isnan(dataset[(i+look_back*sample_per)-1]):
            continue
        
        #Randomly pick a sample
        if np.random.random()>sample:
            continue
        
        #Resample price history
        p=dataset[i:(i+look_back*sample_per)]
        h=npg.aggregate(ix,p,'nanmax',fill_value=np.nan)
        l=npg.aggregate(ix,p,'nanmin',fill_value=np.nan)
        a=np.array([h,l])
        a=(a-p[-1])/(a.max()-a.min())
        a=a.transpose()
        a=a.reshape((1,look_back,2))
        
        #Determine target result - Skip if no signal at 20s
        try:
            #   Find next non-null value
            st=i+look_back*sample_per+np.where(~np.isnan(dataset[i+look_back*sample_per:][:20]))[0][0]
            #   Find last non-null value until period end
            ed=i+look_back*sample_per+np.where(~np.isnan(dataset[i+look_back*sample_per:i+look_back*sample_per+look_ahead*sample_per]))[0][-1]
        except:
            a=np.nan

        if not np.isnan(a).any():
            dataX.append(a[0])
            dataY.append(dataset[st]<dataset[ed])
            idx.append(i)
        
    return np.array(idx), np.array(dataX), np.array(dataY)
        
i, X,Y = create_dataset(df.as_matrix())
idx=[np.array(df.index[ix] for ix in i)]


np.save('i.np',i)
np.save('X.np',X)
np.save('Y.np',Y)
 

#Cutoff train from test
tCutOff=int(X.shape[0]*2/3)
x=X[0:tCutOff]
y=Y[0:tCutOff]

xt=X[tCutOff:]
yt=Y[tCutOff:]


#Extend training set with pair inverse 
ii,xi,yi = create_dataset(1/df.as_matrix()[0:int(df.shape[0]*2/3)])
#xi=xi.reshape(xi.shape[:-1])

x=np.concatenate((x,xi))
y=np.concatenate((y,yi))

np.save('x.np',x)
np.save('y.np',y)

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
from sklearn.metrics import accuracy_score

#Dropout fix
import tensorflow as tf
tf.python.control_flow_ops = tf

def buildModel():
    # create and fit the LSTM network
    #Deep with covolution
    model = Sequential()
    model.add(Convolution1D(input_shape=X.shape[1:],nb_filter=32, filter_length=3, border_mode='same', activation='relu'))  #--Raised to 5 makes difference
    #model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    #model.add(Convolution1D(input_shape=X.shape[1:],nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.2))
    model.add(Flatten())
    #model.add(Dense(64, activation='relu')) #---Not adding Much
    #model.add(Dropout(0.2))                 #---Not adding Much
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def trainModel(maxEpoch,model):
    trainPerf=[]
    testPerf=[]
    
    for i in range(maxEpoch):
        trn=model.fit(x, y, nb_epoch=1, batch_size=64)
        yP=model.predict(xt)
        trainPerf.append([trn.history['acc'][0],trn.history['loss'][0]])
        testPerf.append([accuracy_score(yt, yP>.5),roc_auc_score(yt, yP)])
        
        print('Iteration ',i,testPerf[-1])
    return trainPerf, testPerf

np.random.seed(1664)
model=buildModel()
trainPerf, testPerf=trainModel(2,model)
model.save('models/model.2.ker')

#Prediction
yP=model.predict(x)
sns.distplot(yP)

yP=model.predict(xt)
sns.distplot(yP)

#P&L
#   Treshold Filter
f=abs(yP[:,0]-.5)>.035
ytf=yt[f]

f=(yP[:,0]-.5)>.05
ytf=yt[f]

#   Profitability
c=confusion_matrix(ytf,yP[f,0]>.5)

stake=.535
c.diagonal().sum()*(1-stake) - np.flipud(c).diagonal().sum() * stake

print('Acc=',(c.diagonal().sum()/c.sum()),'AUC=',roc_auc_score(yt, yP))
#Acc= 0.556062813088 AUC= 0.532361017664

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
