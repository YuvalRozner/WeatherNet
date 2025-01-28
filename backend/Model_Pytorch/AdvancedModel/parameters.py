# parametes.py
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device

STATIONS_COORDINATES = {
    'Newe Yaar_data_2005_2024':         (217010, 734820),
    'Tavor Kadoorie_data_2005_2024':    (238440, 734540),
    'Yavneel_data_2005_2024':           (248110, 733730),
}

STATIONS_COORDINATES_COLAB = {
    f'/content/Newe Yaar_data_2005_2024.pkl':         (217010, 734820),
    f'/content/Tavor Kadoorie_data_2005_2024.pkl':    (238440, 734540),
    f'/content/Yavneel_data_2005_2024.pkl':           (248110, 733730),
}

STATIONS_LIST = {
    "Newe Yaar": "186",
    "Tavor Kadoorie": "13"
}

PARAMS = {
    'paths_in_colab': [f'/content/Newe Yaar_data_2005_2024.pkl', f'/content/Tavor Kadoorie_data_2005_2024.pkl'],
    #'fileNames':        ['Newe Yaar_data_2005_2024', 'Tavor Kadoorie_data_2005_2024', 'Yavneel_data_2005_2024'],
    'fileNames':        ['Newe Yaar_data_2005_2024', 'Tavor Kadoorie_data_2005_2024'],
    'target_station':   'Newe Yaar_data_2005_2024',
    'target_station_desplay_name':   'Newe Yaar',
    'target_station_id': 0,
    'epochs' :           20,
    'resume' :           False, # resume train or start new
    'device' :           device,
    'in_channels' :      15 # how many features we have
}

WINDOW_PARAMS = {
    'input_width' :     72, #window input size
    'label_width' :     24, # how many hours to predict to the future
    'shift' :           1,
    'label_columns' :   ['TD (degC)'],
}

LSTM_MODEL_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 2,
}

ADVANCED_MODEL_PARAMS = {
   
    'num_stations':         len(PARAMS['fileNames']),
    'time_steps':           WINDOW_PARAMS['input_width'],
    'feature_dim':          PARAMS['in_channels'],
    'cnn_channels':         16,
    'kernel_size':          3,
    'd_model':              64,
    'nhead':                8,
    'num_layers':           4,
    'target_station_idx':   PARAMS['target_station_id'],
    'label_width':          WINDOW_PARAMS['label_width']
}
