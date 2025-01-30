# parametes.ipynb
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device

STATIONS_COORDINATES = {
    'Tavor Kadoorie':           (238440, 734540),
    'Newe Yaar':                (217010, 734820),
    'Yavneel':                  (248110, 733730),
    'En Hashofet':              (209310, 723170),
    'Eden Farm':                (246190, 708240),
    'Eshhar':                   (228530, 754390),
    'Afula Nir Haemeq':         (226260, 722410)
}

STATIONS_COORDINATES_COLAB = {
    f'/content/drive/MyDrive/final data/Tavor Kadoorie.pkl':     (238440, 734540),
    f'/content/drive/MyDrive/final data/Newe Yaar.pkl':          (217010, 734820),
    f'/content/drive/MyDrive/final data/Yavneel.pkl':            (248110, 733730),
    f'/content/drive/MyDrive/final data/En Hashofet.pkl':        (209310, 723170),
    f'/content/drive/MyDrive/final data/Eden Farm.pkl':          (246190, 708240),
    f'/content/drive/MyDrive/final data/Eshhar.pkl':             (228530, 754390),
    f'/content/drive/MyDrive/final data/Afula Nir Haemeq.pkl':   (226260, 722410)
}

STATIONS_LIST = {
    "Tavor Kadoorie":   "13",
    "Newe Yaar":        "186",
    "Yavneel":          "11",
    "En Hashofet":      "67",
    "Eden Farm":        "206",
    "Eshhar":           "205",
    "Afula Nir Haemeq": "16"
}

PARAMS = {
    #'paths_in_colab': [f'/content/Newe Yaar_data_2005_2024.pkl', f'/content/Tavor Kadoorie_data_2005_2024.pkl'],
    'paths_in_colab': [
        f'/content/drive/MyDrive/final data/Tavor Kadoorie.pkl',
        f'/content/drive/MyDrive/final data/Newe Yaar.pkl',
        f'/content/drive/MyDrive/final data/Yavneel.pkl',
        f'/content/drive/MyDrive/final data/En Hashofet.pkl',
        f'/content/drive/MyDrive/final data/Eden Farm.pkl',
        f'/content/drive/MyDrive/final data/Eshhar.pkl',
        f'/content/drive/MyDrive/final data/Afula Nir Haemeq.pkl'],
    'fileNames':        ['Tavor Kadoorie', 'Newe Yaar', 'Yavneel', 'En Hashofet', 'Eden Farm', 'Eshhar', 'Afula Nir Haemeq'],
    'target_station':   'Tavor Kadoorie',
    'target_station_desplay_name':   'Tavor Kadoorie',
    'target_station_id': 0,
    'device' :           device,
    'in_channels' :      15 # how many features we have
}

WINDOW_PARAMS = {
    'input_width' :     72, #window input size
    'label_width' :     12, # how many hours to predict to the future
    'shift' :           1,
    'label_columns' :   ['TD (degC)'],
}

TRAIN_PARAMS = {
    'epochs' :          50,  
    'batch_size':       32,
    'lr':               1e-5,    
    'resume':           False,
    'device':           PARAMS['device'],
    'early_stopping_patience':10,
    'scheduler_patience':3,
    'scheduler_factor':  0.5,
    'min_lr':            1e-7
}

INFERENCE_PARAMS = {
    'params_path':            ['output/parameters.py'],
    'weights_paths':          ['output/checkpoints/best_checkpoint.pth'],
    'scaler_folder_path':     'output/scalers',
    'analyze_output_folder_per_folder':  ['output/analyze_output'],
    'analyze_output_folder':  'output/analyze_output',

}
ADVANCED_MODEL_PARAMS = {
    'num_stations':         len(PARAMS['fileNames']),
    'time_steps':           WINDOW_PARAMS['input_width'],
    'feature_dim':          PARAMS['in_channels'],
    'cnn_channels':         15,
    'kernel_size':          3,
    'd_model':              64,
    'nhead':                8,
    'num_layers':           4,
    'target_station_idx':   PARAMS['target_station_id'],
    'label_width':          WINDOW_PARAMS['label_width']
}

