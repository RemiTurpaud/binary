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
import numpy_groupies as npg

HistDepth=60

#Build training dataframe
# convert [h,l,c] into 2D series. Centered on last close
def createDatasetWeek(dataset, look_back=60,look_ahead=3,sample_per=60,sample=.2):
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


def createDatasetWeek():
    uniqueWeek=set(zip(df.index.year,df.index.week))
    
    i = np.array([])
    X = np.array([]).reshape((0,HistDepth,2))
    Y = np.array([])
    
    for w in uniqueWeek:
        ii, x,y = createDatasetWeek(df[(df.index.year==w[0]) & (df.index.week==w[1])].as_matrix(),look_back=HistDepth)
        i=np.append(i,ii,0)
        X=np.append(X,x,0)
        Y=np.append(Y,y,0)
    
    
    i, X,Y = createDatasetWeek(df.as_matrix(),sample=.2)
    i=df.index[i]
    
    np.save('data/i',i)
    np.save('data/X',X)
    np.save('data/Y',Y)
 
#Load data 
i=np.load('data/i.npy')
X=np.load('data/X.npy')
Y=np.load('data/Y.npy')

#Cutoff train from test
tCutOff=int(X.shape[0]*2/3)
x=X[0:tCutOff]
y=Y[0:tCutOff]

xt=X[tCutOff:]
yt=Y[tCutOff:]

del X,Y,i

print('Data Loaded - Start training')

#Test
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Reshape
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
def buildModel(convol,cdropout,hidden,hdropout):
    # create and fit the LSTM network
    #Deep with covolution
    model = Sequential()
    model.add(Reshape(X.shape[1:],input_shape=x.shape[1:]))
    for c, d in zip(convol,cdropout):
        model.add(Convolution1D(filters=int(c), kernel_size=5, padding='same', activation='relu'))  #--Raised to 5 makes difference
        if d>0:
            model.add(Dropout(d))

    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())

    for h, d in zip(hidden,hdropout):
        model.add(Dense(int(h), activation='relu'))
        if d>0:
            model.add(Dropout(d))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def trainModel(maxEpoch,model):
    trainPerf=[]
    testPerf=[]
    
    for i in range(maxEpoch):
        trn=model.fit(x, y, epochs=5, batch_size=64,verbose=0)
        yP=model.predict(xt)
        trainPerf.append([trn.history['acc'][0],trn.history['loss'][0]])
        testPerf.append([accuracy_score(yt, yP>.5),roc_auc_score(yt, yP),(sum(yP>.5)/len(yP))[0]])
        
        print('Iteration ',i,testPerf[-1])
    return trainPerf, testPerf

def saveModel(model,mid=''):
        # serialize model to JSON
    model_json = model.to_json()
    with open("models/model."+str(mid)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/model."+str(mid)+".h5")
    print("Saved model to disk")

def evalModels(params):
    f= open("./models/modelPerf.csv","a")
    for i in range(len(params)):
        np.random.seed(1664)
        model=buildModel(params[i][0],params[i][1],params[i][2],params[i][3])
        trainPerf, testPerf=trainModel(2,model)
        f.write(','.join([str(i),str(trainPerf),str(testPerf)])+'\n')
        saveModel(model,i)
    f.close()

#Evaluate models
params=[]
params.append([[16,16],[.1,.1],[4],[.1]])
params.append([[16,16],[.2,.2],[4],[.1]])
params.append([[32,16],[.1,.1],[4],[.1]])
params.append([[16,8],[.1,.1],[4],[.1]])
params.append([[16,8],[.1,.1],[8,4],[.1,.1]])
params.append([[16,8],[.01,.01],[8,4],[.01,.01]])
params.append([[16,8],[.1,.1],[8,8,4],[.1,.1,.1]])

evalModels(params)
