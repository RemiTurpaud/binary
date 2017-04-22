# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:26:44 2017

Test methods to override the bin.com trading environment with historical quotes

@author: remi
"""

import numpy as np 
import pandas as pd 

#Epoch, Ask, Price
eap=np.loadtxt(open("data/binCom-live.csv", "rb"), delimiter=",", skiprows=1)
HistDepth=60
#Global buffers
Tick=0

def getQuote(ws,apiUrl,verbose=False): 

    global Tick
    
    #Add to buffer
    bufferAdd(eap[Tick,2],eap[Tick,1],eap[Tick,0],'')

    Tick +=1
    
    if verbose:
        print([int(m['spot_time']),float(m['ask_price']),float(m['spot'])])
    
    return 1

def trade(ws,apiUrl):
    global Tepoch, Tcontracts,Ttrades

    Ttrades.append(Bprice[-1])
    Tepoch.append(Bepoch[-1])
    
    print(Tepoch[-1],'Contract bought -$',Bask[-1])
        
    return 1

def strategy():
    #Only trade if profitable
    if (Amount-Bask[-1])/Bask[-1] <.87:
        return 0

    #Only trade if the previous contract is over
    if Bepoch[-1]-Tepoch[-1]<Duration:
        return 0

    #Check if no missing history point        
    if np.isnan(Bprice).any():
        return False


    #Calculate h,l
    ix=np.round((Bepoch[-1]-Bepoch)/60).astype('int')
    
    #   Check if we have collected enough history
    if max(ix)-min(ix)<HistDepth-1:
       return 0

    #Check if we have a value for all bars
    if np.diff(npg.aggregate(ix,ix,'max')[[-HistDepth,-1]])[0]!=HistDepth-1:
        print(Bepoch[-1],': Incomplete time series')
        return 0
        
    h=npg.aggregate(-ix+max(ix),Bprice,'max',fill_value=np.nan)[-HistDepth:]
    l=npg.aggregate(-ix+max(ix),Bprice,'min',fill_value=np.nan)[-HistDepth:]
    a=np.array([h[-HistDepth:],l[-HistDepth:]])
    a=(a-Bprice[-1])/(a.max()-a.min())

    a=a.transpose()
    a=a.reshape((1,HistDepth,2))

    #a=a.reshape((1,2, HistDepth))
    y=model.predict_on_batch([a])[0]
        
    return y>.535

#-----------------Main Loop    

from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score

#model=load_model('models/model.2.ker')

ws,apiUrl=0,0

Tick=0
bufferReset(ws,apiUrl,True)
while Tick<len(eap) and getQuote(ws,apiUrl)>=0:
    if strategy():
        trade(ws,apiUrl)
    #time.sleep(.001)

#Check performance
res=[]
for p,e in zip(Ttrades, Tepoch[1:]):
    #Get the next timestamp - the traded one
    ne=eap[eap[:,0]>=e+1,0][0]
    res.append(eap[eap[:,0]>=ne+3*60-1,2][0]>eap[eap[:,0]>=ne,2][0])

np.mean(res)
sum(res)
len(res)

p,e = Ttrades[-4], Tepoch[-4]

e=1491872255
p=eap[eap[:,0]==e,2][0]

Ttrades[1491876286==Tepoch[1:]]

eap[1491876286==eap[:,0]]

eap[eap[:,0]==e+4,:][0:3]