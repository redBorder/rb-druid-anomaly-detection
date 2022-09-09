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
from fastapi import FastAPI, Request
from pydantic import BaseModel

import json
from types import SimpleNamespace

import requests


def build_aggregations(data_aggregations):
  aggregations={}
  for aggregation in data_aggregations:
    aggregations.update({aggregation["name"]: {"type": aggregation["type"], "fieldName": aggregation["fieldName"]}})
  return aggregations

def build_filter(filters):
  druid_filter = filters
  return Dimension(druid_filter["dimension"]) == druid_filter["value"]

def query_druid(broker_host, broker_port, data_sources, granularity, intervals, aggregations, filters):
  """
    Returns: 
      data: panda table with result from druid
  """
  druid_url = "http://{}:{}".format(broker_host, broker_port)
  print("druid_url {}".format(druid_url))
  query = PyDruid(druid_url, 'druid/v2')

  # TODO: Check query type, and what happen when there is more than one filter
  ts = query.timeseries(
      datasource=data_sources,
      granularity=granularity,
      intervals=intervals,
      aggregations=aggregations,
      filter=filters
  )

  # get the data
  data = query.export_pandas()
  return data

def build_predictions(training_data, new_data):
   # init setup
  s=setup(training_data,normalize = True,silent=True)
  # train model
  #model = create_model('iforest') 
  model = create_model('knn') 
  # assign anomaly labels on training data
  model_results = assign_model(model)
  print(model_results)
  
  # prediction 
  predictions = predict_model(model, new_data)
  print("predictions.head(): ")
  print(predictions.head())

  print("predictions: ")
  print(predictions)

  # save model pipeline
  save_model(model, 'model_pipeline')
  return predictions

def build_anomalies(predictions, aggregation):
  """ Create anomalies json response with predictions generated from the model.
    Args:
      predictions

    Returns:
      anomalies
  """
  df = predictions
  df = df.reset_index()
  anomalies = []

  predicted_anomalies=predictions[predictions['Anomaly']==1]

  for index,row in predicted_anomalies.iterrows():
    print(row[aggregation], row['timestamp'], row['Anomaly'])
    anomaly =  { 
      "timestamp": row['timestamp'], 
      "expected" : row[aggregation]
    }
    print(anomaly)
    anomalies.append(anomaly)

  print("Anomalies: ")
  print(anomalies)
  return anomalies

app = FastAPI()

@app.get("/")
async def root():
  return "Welcome to Redborder Druid Anomaly Detector! Documentation: https://github.com/redBorder/rb-druid-anomaly-detection"

@app.post("/anomaly_detection/")
#async def anomaly_detection(data_request: DataRequest) -> DataRequest:    
async def anomaly_detection(request: Request):
    data_request = await request.json() 
  
    status = True
    error = ''
    anomalies = []
   
    if not error:
      try: 
        training_data  = query_druid(data_request["broker_host"], data_request["broker_port"], data_request["data_sources"], data_request["granularity"], data_request["training_intervals"], build_aggregations(data_request["aggregations"]), build_filter(data_request["filters"]))
        print("Training data: ")
        print(training_data)
        new_data       = query_druid(data_request["broker_host"], data_request["broker_port"], data_request["data_sources"], data_request["granularity"], data_request["intervals"], build_aggregations(data_request["aggregations"]), build_filter(data_request["filters"]))
        print("New data: ")
        print(new_data)
      except:
        error = 'Internal Error - Making druid queries'

    if not error:
      try:
        predictions = build_predictions(training_data, new_data)      
      except:
        error = 'Internal Error - Making predictions'

    if not error: 
      try:
        anomalies = build_anomalies(predictions, data_request["aggregation"])
      except:
        error = 'Internal Error - Processing anomalies'
    
    if error:
      anomalies = []
      status = False
    
    return json.loads(json.dumps({ "status" : status, "error" : error, "anomalies" : anomalies }))
