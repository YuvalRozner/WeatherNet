# Simple Model - Plan
## thouths to explore:
- what the data representing after 1d cnn each feature
    - in my understanding it is representing short term patters acuurs beacuse the idea it to 
    input to the id cnn a window.
    - but - how the transformer can deal with such dat
- maybe do a 1d cnn model by it own to test and transformer model by it own.
- how to use the transformer model with time series data.


## Description:
- in this example we will do a simple model to predict the target station temperature.
- it is going to be only with one station data. 

## Input Data:
- only the target station data.
- time series data with 1 hour frequency.
- dataset includes meteorological features (Rain, RH, TD, TDmax, TDmin, WD, WDmax, WS, WSmax, etc.) with cyclical time encodings (e.g., sin and cos for day and year)

## The Model:
1. 1D CNN to extract local features of the region
2. multihead transformer for long term dependencies.

## Goals to Achieve:
1. window generator class (python file).
2. model class (python file).
3. train (python file).
4. inference (python file).
5. evaluation. ( need to investigate how to evaluate the model.)
    - normal evaluation ? 
    - per base model ?
6. visualization. ( need to investigate what can i show? )
    - the layer of the model ?
    - the cnn learned features maps ? 
    - the transformer attention connections ?
7. saving and loading the model options.
    - save and load for continue training.
    - save and load for inference.
8. hyperparameters tuning.