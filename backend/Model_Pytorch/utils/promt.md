i am doing a project tring to forcast temperature at station using deep learning.
the arcitucture:

deep learning model to forecast temperature at a target weather station using historical data from several nerby stations. 
Each station’s dataset includes meteorological features (Rain, RH, TD, TDmax, TDmin, WD, WDmax, WS, WSmax, etc.) with cyclical time encodings (e.g., sin and cos for day and year) and its location in "New Network of Israel" coordinates in a defrent file. 

**Desired Architecture:**

1. **Per-Station Feature Extraction:**  
   - CNN to extract short-term temporal features from each station’s time series.
   - pooling layer

2. **location to each station**
   - since we have each station location in "New Network of Israel" coordinates we want to include this before the transformer

2. **multi-head Transformer:**  
   - Feed the extracted features into a multi-head Transformer to capture spatial dependencies between stations.
   - long term dependencies

4. **Prediction:**  
   - Use a final dense layer to predict the target station’s temperature.

This architecture leverages both spatial and temporal dependencies by integrating meteorological data, time cyclic encodings, and precise geographical positions.

other instructions:
   - Train the model using a sliding window approach over the time series data.
   - pythorch architecture

