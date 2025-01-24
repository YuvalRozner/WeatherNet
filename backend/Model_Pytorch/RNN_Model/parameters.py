# parametes.py
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device

PARAMS = {
    'fileName' : 'data.csv',
    'filePah' : "..\\common\\jena_climate_2009_2016.csv",
    'epochs' : 20,
    'device' : device
}

WINDOW_PARAMS = {
    'input_width' : 24,
    'label_width' : 1,
    'shift' : 1,
    'label_columns' : ['T (degC)'],
}

LSTM_MODEL_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 2,
}