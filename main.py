#import urllib.request

import pandas as pd

from pylab import plt
from pydruid import *
from pydruid.client import *
from pydruid.query import QueryBuilder
from pydruid.utils.postaggregator import *
from pydruid.utils.aggregators import *
from pydruid.utils.filters import *

import pycaret
from pycaret.anomaly import *

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

import json

def process_rb(data_request, druid_query):
  druid_url = "http://{}:{}".format(data_request.brokerHost, data_request.brokerPort)
  print("druid_url {}".format(druid_url))
  query = PyDruid(druid_url, 'druid/v2')
  
  # training
  ts = query.timeseries(
      datasource=druid_query["dataSource"],
      granularity=data_request.granularity,
      intervals="%s/p1d" % (druid_query["granularity"]["origin"], ),
      aggregations={'bytes': doublesum('sum_bytes')},
      filter=Dimension('sensor_name') == 'ASR'
  )
  
  # get the data
  data = query.export_pandas()
  
  print(data)
  
  # init setup
  s=setup(data,normalize = True,silent=True)
  
  # train isolation forest model
  iforest = create_model('iforest')
  
  # assign anomaly labels on training data
  iforest_results = assign_model(iforest)
  
  #iforest_results[iforest_results['Anomaly']==1].shape
  #iforest_results[iforest_results['Anomaly']==0].shape
  
  print(iforest_results)
  
  # prediction
  ts = query.timeseries(
      datasource=druid_query["dataSource"],
      granularity=data_request.granularity,
      intervals=druid_query["intervals"],
      aggregations={'bytes': doublesum('sum_bytes')},
      filter=Dimension('sensor_name') == 'ASR'
  )
  new_data = query.export_pandas()
  
  print(new_data)
  
  predictions = predict_model(iforest, new_data)
  
  print(predictions)
  
  # save iforest pipeline
  save_model(iforest, 'iforest_pipeline')
  return predictions

# From the predictions that returns the model
# build a json to reply to rb-manager with 
# an json event per anomaly and its predicted value
def process_anomalies(predictions):
  df = predictions
  df = df.reset_index()
  anomalies = []

  for index,row in df.iterrows():
    if row['Anomaly'] == 1 :
      print(row['bytes'], row['timestamp'], row['Anomaly'])
      anomaly =  { 
        "timestamp": row['timestamp'], 
        "expected" : row['bytes']
      }
      print(anomaly)
      anomalies.append(anomaly)

  print("Anomalies: ")
  print(anomalies)
  return anomalies

####### REST API ##########
class Data(BaseModel):
  query: str
  granularity: str
  granularityRange: str
  timeseriesRange: int
  frequency: str
  queryEndTimeText: str
  detectionWindow: int
  hoursOfLag: str
  clusterId: str
  tsFramework: str
  adModels: str
  sigmaThreshold: int
  tsModels: str
  brokerHost: str
  brokerPort: int


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process_query/")
async def process_query(data_request: Data) -> Data:    
    response = {}
    try:
      druid_query = json.loads(data_request.query)
      predictions = process_rb(data_request, druid_query)
      anomalies = process_anomalies(predictions)
    except:
      response = {
        "status" : "fail",
        "error" : "Error processing query",
        "anomalies" : []
      }
    else:
      response = {
        "status" : "success",
        "error" : "false",
        "anomalies" : anomalies
      }
    return json.loads(json.dumps(response))
