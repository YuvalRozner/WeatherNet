import torch
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device

# for training - where the output will be saved
file_path = os.path.dirname(__file__)

# you need to put int inference_base_path, folders where each folder has the model files - inference_base_path/parameters.py, inference_base_path/output/scalers, inference_base_path/output/checkpoints. 
inference_base_path = os.path.dirname(__file__)
models_paths_dir_names =  ['model_1'] # for instance for alot of models we want to inference: ['model_1','model_2' ... ] for one : ['model_1']


output_path = os.path.join(file_path, 'output')
checkpoints_path = os.path.join(output_path, 'checkpoints')
scalers_path = os.path.join(output_path, 'scalers')
inference_output_path = os.path.join(output_path, 'inference_output')

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
    'in_channels' :      15, # how many features we have
    'output_path':       output_path,
    'checkpoints_path':  checkpoints_path,
    'scalers_path':      scalers_path,
    'inference_output_path': inference_output_path
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
    'checkpoint_dir' :  PARAMS['checkpoints_path'],  
    'resume':           False,
    'device':           PARAMS['device'],
    'early_stopping_patience':10,
    'scheduler_patience':3,
    'scheduler_factor':  0.5,
    'min_lr':            1e-7
}

ADVANCED_MODEL_PARAMS = {
    'num_stations':         len(PARAMS['fileNames']),
    'time_steps':           WINDOW_PARAMS['input_width'],
    'feature_dim':          PARAMS['in_channels'],
    'kernel_size':          3,
    'd_model':              64,
    'nhead':                8,
    'num_layers':           4,
    'target_station_idx':   PARAMS['target_station_id'],
    'label_width':          WINDOW_PARAMS['label_width'],
    'output_per_feature':   3,
    'use_batch_norm':       False,
    'use_residual':         False
}

models_paths_dir_names_full_paths = [os.path.join(inference_base_path, model_folder_name) for model_folder_name in models_paths_dir_names]

INFERENCE_PARAMS = {
    'params_path':             [os.path.join(folder, 'parameters.py') for folder in models_paths_dir_names_full_paths],
    'weights_paths':           [os.path.join(folder, 'checkpoints', 'best_checkpoint.pth') for folder in models_paths_dir_names_full_paths],
    'scaler_folder_path':      PARAMS['scalers_path'],
    'inference_output_path_per_model':  models_paths_dir_names_full_paths, # for saving the output of the inference in the model folder for each model
    'inference_output_path':  os.path.join(inference_base_path, 'analyze_output'), # for saving the output of the inference of all models in one folder (later analyze.py will use it)
}