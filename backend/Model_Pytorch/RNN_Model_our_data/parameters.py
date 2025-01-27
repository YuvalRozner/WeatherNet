# parametes.py
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device

PARAMS = {
    'fileName' : 'Newe Yaar_data_2005_2024',
    'epochs' : 20,
    'resume' : False, # resume train or start new
    'device' : device,
    'in_channels' : 15 # how many features we have
}

WINDOW_PARAMS = {
    'input_width' : 72, #window input size
    'label_width' : 24, # how many hours to predict to the future
    'shift' : 1,
    'label_columns' : ['TD (degC)'],
}

LSTM_MODEL_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 2,
}