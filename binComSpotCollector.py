# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:26:44 2017

@author: remi
"""

import websocket
import json
import time

#os.chdir('./remi/ML/binary')
f= open("./binCom-eurusdSpot.csv","a")

json_data = json.dumps(
{
  "proposal": 1,
  "amount": "100",
  "basis": "payout",
  "contract_type": "CALL",
  "currency": "USD",
  "duration": "60",
  "duration_unit": "s",
  "symbol": "frxEURUSD"
})


def getQuote(ws,apiUrl,verbose=False): 

    #Reconnect if disconnected
    if not ws.connected:
        print('Reconnect websocket')
        ws.connect(apiUrl)

    try:   
        ws.send(json_data)
        message =  ws.recv()
        m=json.loads(message)
    except:
        print('Error communicating to web socket')
        return 0
    
    if 'error' in m:
        print('Error:',m['error'])
        
        if 'message' in m['error']:
            if 'This market is presently closed' in m['error']['message']:
                print('Market Closed - Closing')
                return -1
        return 0
    try:
        m=m['proposal']
    except:
        print('No conract proposal:',message)
        return 0
            
    f.write(','.join([str(m['spot_time']),str(m['ask_price']),str(m['spot'])])+'\n')
    if int(m['spot_time'])%100 ==0:
        f.flush()
    if verbose:
        print([int(m['spot_time']),float(m['ask_price']),float(m['spot'])])
    
    return 1
    
apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
ws = websocket.create_connection(apiUrl)

while getQuote(ws,apiUrl)>=0:
    time.sleep(1)
    
ws.close()
f.close()
