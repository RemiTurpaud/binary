# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:29:32 2017

@author: remi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("darkgrid")

df=pd.DataFrame.from_csv('data/binCom-eurusdSpot.csv')

t=df[[1,]]
p=df[[0,]]

t.index=pd.to_datetime(t.index,unit='s')
p.index=pd.to_datetime(t.index,unit='s')

plt.plot(t)
#plt.plot(p)

#----predict
t.columns=['']
ts=t.resample('1Min').ohlc()
ts.columns=ts.columns.droplevel()

#---Price to enter
ps=p.resample('1Min').ohlc()
ps.columns=ps.columns.droplevel()

fp=np.array(ps.close<=53.5)

#Determine call at 1 mn
ts['t']=np.nan
ts.t=ts['close'].shift(-1, freq='min')
ts.t=ts.t>ts.close

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
        
X,Y = create_dataset(np.matrix(ts[['high','low','close','t']]))

X=X.reshape(X.shape[:-1])

#Remove NaN
#n=~np.any(np.isnan(X), axis=(1,2))

n=np.apply_along_axis(np.sum,1,X)
n=np.apply_along_axis(np.sum,1,n)
n=np.isnan(n)==False
X=X[n]
Y=Y[n]

from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score
import tensorflow as tf
tf.python.control_flow_ops = tf

model=load_model('models/model.ker')
yP=model.predict(X)

#P&L
#   Treshold Filter
#f=abs(yP[:,0]-.5)>.035
f=(yP[:,0]-.5)>.035
f=f&fp[60:-1]

ytf=Y[f]


#   Profitability
c=confusion_matrix(ytf,yP[f,0]>.5)

stake=.535

print('Acc=',(c.diagonal().sum()/c.sum()),'AUC=',roc_auc_score(Y, yP))
#Acc= 0.557811562312 AUC= 0.53421291115

#   Equity timeline
r=(1-stake)*((yP[:,0]>.5)==Y)
r[(yP[:,0]>.5)!=Y]=-stake
r[np.invert(f)]=0

stake=ps.close[60:-1].values/100

r=(1-stake)*((yP[:,0]>.5)==Y)
r[(yP[:,0]>.5)!=Y]=-stake[(yP[:,0]>.5)!=Y]
r[np.invert(f)]=0

sum(r)
plt.plot(r.cumsum())
plt.plot((r/100+1).cumprod())

cr=r.cumsum()
plt.plot((t-t.min())/(t.max()-t.min()))
plt.plot((cr-cr.min())/(cr.max()-cr.min()))