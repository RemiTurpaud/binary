# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:26:44 2017

@author: remi
"""

import websocket
import json
import datetime

#------------------On-line
M=[]
T=[]

def on_open(ws):
    json_data = json.dumps({'ticks':'frxEURUSD'})
    ws.send(json_data)

def on_message(ws, message):
    #print('ticks update: %s' % message)
    print('Tick = '+json.loads(message)['tick']['quote'])
    T.append([datetime.datetime.now(),json.loads(message)['tick']['epoch'],json.loads(message)['tick']['quote']])
    M.append(message)

def getQuotes():
    apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    ws = websocket.WebSocketApp(apiUrl, on_message = on_message, on_open = on_open)
    ws.run_forever()

getQuotes()    


#------------------Ad-hoc: request for contract
json_data = json.dumps(
{
  "proposal": 1,
  "amount": "100",
  "basis": "payout",
  "contract_type": "CALL",
  "currency": "USD",
  "duration": "300",
  "duration_unit": "s",
  "symbol": "frxEURUSD"
})

apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
ws = websocket.create_connection(apiUrl)
ws.send(json_data)
result =  ws.recv()
ws.close()


m=json.loads(result)['proposal']
m['spot_time'],m['ask_price'],m['spot']

#------------------Ad-hoc: blind trade order 
json_auth = json.dumps(
{
  "authorize": "UmU3O7unB621krc"
})

json_buy = json.dumps(
{
  "buy": 1,
  "price": "5.35",
  "parameters":{
      "amount": "10",
      "basis": "payout",
      "contract_type": "CALL",
      "currency": "USD",
      "duration": "60",
      "duration_unit": "s",
      "symbol": "frxEURUSD"
  }
})


apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
ws = websocket.create_connection(apiUrl)
ws.send(json_auth)
authResult =  ws.recv()

ws.send(json_buy)
buyResult =  ws.recv()

ws.close()


#------------------Ad-hoc: trade order from proposal
json_auth = json.dumps(
{
  "authorize": "UmU3O7unB621krc"
})

json_req = json.dumps(
{
  "proposal": 1,
  "amount": "1",
  "basis": "payout",
  "contract_type": "CALL",
  "currency": "USD",
  "duration": "60",
  "duration_unit": "s",
  "symbol": "frxEURUSD"
})


apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
ws = websocket.create_connection(apiUrl)
ws.send(json_auth)
authResult =  ws.recv()

ws.send(json_req)
propResult =  ws.recv()

m=json.loads(propResult )['proposal']

json_buy = json.dumps(
{
  "buy": m['id'],
  "price": 10
})

ws.send(json_buy)
buyResult =  ws.recv()

json.loads(buyResult )['buy']

ws.close()
