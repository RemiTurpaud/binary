# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:26:44 2017

@author: remi
"""

import websocket
import json
import time
import numpy as np
import numpy_groupies as npg

#os.chdir('./remi/ML/binary')
f= open("./data/binCom-live.csv","a")

#Contract details
Amount=5
AmtMax=Amount*(5.25/5)
MinTradePer=30      #Minimum period (s) between two trades

ContractType="CALL"
Duration=3*60

#Model parameters
HistDepth=60

#Global buffers
Blen=int(60*60*1.2)
Bprice=np.full(Blen,np.nan,dtype='float32')
Bask=np.full(Blen,np.nan,dtype='float32')
Bepoch=np.full(Blen,np.nan,dtype='int32')
ContractId=''

#Trading State
Tepoch=[0]
Tcontracts=['']
Ttrades=[]
TtradesConf=[]

Symbol="frxEURUSD"

jReqQuote = json.dumps(
{
  "proposal": 1,
  "amount": Amount,
  "basis": "payout",
  "contract_type": ContractType,
  "currency": "USD",
  "duration": Duration,
  "duration_unit": "s",
  "symbol": Symbol
})

jReqHist = json.dumps({
  "ticks_history": Symbol,
  "end": "latest",
  "count": Blen
})

jAuth = json.dumps(
{
  "authorize": "UmU3O7unB621krc"
})


def bufferAdd(price,ask,epoch,contract):
    global Bprice,Bask,Bepoch,ContractId
    Bprice=np.append(Bprice[1:],float(price))
    Bask=np.append(Bask[1:],float(ask))
    Bepoch=np.append(Bepoch[1:],int(epoch))

    ContractId=contract

    #Save to file            
    f.write(','.join([str(epoch),str(ask),str(price)])+'\n')
    if int(epoch)%100 ==0:
        f.flush()

def bufferReset(ws,apiUrl,empty=False):
    global Bprice,Bask,Bepoch,ContractId,Tepoch,Tcontracts,Ttrades,TtradesConf
    Bprice=np.full(Blen,np.nan,dtype='float32')
    Bask=np.full(Blen,np.nan,dtype='float32')
    Bepoch=np.full(Blen,np.nan,dtype='int32')
    ContractId=''
    
    #Fill array with history
    if not empty:
        #Request quote        
        checkConnect(ws,apiUrl)
        ws.send(jReqHist)
        message =  ws.recv()
        m=json.loads(message)
        
        #Add to buffer
        Bprice=np.array(m['history']['prices'],dtype='float32')
        Bepoch=np.array(m['history']['times'],dtype='int32')

    #Reset the trading buffer
    Tepoch=[0]
    Tcontracts=['']
    Ttrades=[]
    TtradesConf=[]

def checkConnect(ws,apiUrl,force=False):
      #Reconnect if disconnected
    if not ws.connected or force:
        print('Reconnect websocket')
        ws.connect(apiUrl)
        #Authenticate
        ws.send(jAuth)
        authResult =  ws.recv()

def getTick(ws,apiUrl,bars=1): 
    
    jReqTick = json.dumps(
    {
      "ticks_history": Symbol,
      "end": "latest",
      "count": bars
    })
    #Connection check
    checkConnect(ws,apiUrl)

    #Request quote
    try:   
        ws.send(jReqTick)
        message =  ws.recv()
        m=json.loads(message)
        
        #Add to buffer
        bufferAdd(float(m['history']['prices'][0]),np.nan,int(m['history']['times'][0]),np.nan)

    except:
        print('Error collecting tick')
        return 0
    
    return 1

def getQuote(ws,apiUrl,verbose=False): 

    #Connection check
    checkConnect(ws,apiUrl)

    #Request quote
    try:   
        ws.send(jReqQuote)
        message =  ws.recv()
        m=json.loads(message)
    except:
        print('Error communicating to web socket - try to reconnect')
        checkConnect(ws,apiUrl,True)
        return 0

    #Check error and market close    
    if 'error' in m:
        print('Error:',m['error'])
        
        #Try to get tick
        getTick(ws,apiUrl)
        
        if 'message' in m['error']:
            if 'This market is presently closed' in m['error']['message']:
                print('Market Closed - Closing')
                return -1
        return 0
        
    #Check proposal
    if 'proposal' in m:
        m=m['proposal']
    else:
        print('No conract proposal:',m)
        return 0

    #Add to buffer
    bufferAdd(m['spot'],m['ask_price'],m['spot_time'],m['id'])

    if verbose:
        print([int(m['spot_time']),float(m['ask_price']),float(m['spot'])])
    
    return 1

def trade(ws,apiUrl):
    global Tepoch, Tcontracts,Ttrades
    #Check connection
    checkConnect(ws,apiUrl)
    
    #Only trade if the previous contract is over
    if Bepoch[-1]-Tepoch[-1]<Duration:
        return 0
    
    #Issue buy order
    json_buy = json.dumps(
    {
      "buy": ContractId,
      "price": AmtMax    
    })

    try:    
        ws.send(json_buy)
        buyResult =  ws.recv()
    except:
        print('Unable to send/recieve buy request')
        return 0

    #Check result        
    m=json.loads(buyResult)
    if 'error' in m:
        print('Error buying contract:',m)  #m['error'])
        return 0
    else:
        Ttrades.append(m['buy'])
        Tepoch.append(m['buy']['start_time'])
        Tcontracts.append(m['buy']['contract_id'])
        print(Tepoch[-1],'Contract bought -$',Amount)
        
    return 1

def strategy():
    global TtradesConf
    
    #Only trade if profitable
    if (Amount-Bask[-1])/Bask[-1] <.87:
        return 0

    #Only trade if the previous contract is over
    if Bepoch[-1]-Tepoch[-1]<MinTradePer:
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

    a=a.reshape((1,HistDepth, 2))
    y=model.predict_on_batch([a])[0]

    if y>.535:
        TtradesConf.append(y[0])
        print(Bepoch[-1],':Order. Conf=',y)
    return y>.535

#-----------------Main Loop    
#Load model
#from keras.models import load_model
from keras.models import model_from_json
#model=load_model('models/model.3.ker')
#Load model
json_file = open('models/model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("models/model.h5")
print("Loaded model from disk")

#Define connection
apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
ws = websocket.create_connection(apiUrl)
#Authenticate
ws.send(jAuth)
authResult =  ws.recv()


bufferReset(ws,apiUrl)
t=time.time()
while getQuote(ws,apiUrl)>=0:
    if strategy():
        trade(ws,apiUrl)
    time.sleep(max(1.001-(time.time()-t),0))
    t=time.time()

f.close()    
