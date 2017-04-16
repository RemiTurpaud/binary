# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:26:44 2017

@author: remi
"""

import websocket
import json
import datetime

#M=[]
f= open("./binCom-eurusd.csv","w+")

def on_open(ws):
    json_data = json.dumps({'ticks':'frxEURUSD'})
    ws.send(json_data)

def on_message(ws, message):
#    print('Tick = '+json.loads(message)['tick']['quote'])
#    M.append(message)
    f.write(','.join([str(datetime.datetime.now()),str(json.loads(message)['tick']['epoch']),str(json.loads(message)['tick']['quote'])])+'\n')
    if int(json.loads(message)['tick']['epoch'])%100 ==0:
        f.flush()


def getQuotes():    

    apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    ws = websocket.WebSocketApp(apiUrl, on_message = on_message, on_open = on_open)
    ws.run_forever()

getQuotes()
f.close()

