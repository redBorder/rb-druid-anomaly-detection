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

def get_aggregations(druid_query):
  aggregations={}
  for aggregation in druid_query["aggregations"]:
    aggregations.update({aggregation["name"]: {"type": aggregation["type"], "fieldName": aggregation["fieldName"]}})
  return aggregations

def get_filter(druid_query):
  druid_filter = druid_query["filter"]
  return Dimension(druid_filter["dimension"]) == druid_filter["value"]

def process_rb(data_request, druid_query):
  druid_url = "http://{}:{}".format(data_request.broker_host, data_request.broker_port)
  print("druid_url {}".format(druid_url))
  query = PyDruid(druid_url, 'druid/v2')
  
  print("Interval are: ")
  print("%s/p1d" % (druid_query["granularity"]["origin"], ))
  # training
  # TODO: Check query type, and what happen when there is more than one filter
  ts = query.timeseries(
      datasource=druid_query["dataSource"],
      granularity=data_request.granularity,
      intervals="%s/p1d" % (druid_query["granularity"]["origin"], ),
      aggregations=get_aggregations(druid_query),
      filter=get_filter(druid_query)
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

def process_anomalies(predictions):
  """ Create anomalies json response with predicctions generated from the model.
    Args:
      predictions

    Returns:
      anomalies
  """
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
class DataRequest(BaseModel):
  query: str
  granularity: str
  granularity_range: str
  timeseries_range: int
  frequency: str
  query_end_time_text: str
  detection_window: int
  hours_of_lag: str
  broker_host: str
  broker_port: int

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process_query/")
async def process_query(data_request: DataRequest) -> DataRequest:    
    response = {}
    
    druid_query = json.loads(data_request.query)
    try:    
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
