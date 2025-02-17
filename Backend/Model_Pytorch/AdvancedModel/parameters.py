import torch
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device

# must define these 3 variables below!!!
###########################################################################################################################################################

# for training - where the output will be saved
#base_path = 'C:\\Users\\dorsh\\Documents\\GitHub\\WeatherNet\\backend\\Model_Pytorch\\AdvancedModel\\models' # in general we put the folder of the path that contains the parameters.py file
local_path = 'C:\\Users\\dorsh\\Documents\\GitHub\\WeatherNet\\backend\\Model_Pytorch\\AdvancedModel\\models' # in general we put the folder of the path that contains the parameters.py file
colab_path = r'/content/models_for_inference'
#'C:\\Users\\dorsh\\Documents\\GitHub\\WeatherNet\\backend\\Model_Pytorch\\AdvancedModel\\models'
name_of_the_model_to_save_train = r'1_12'

# for inference
models_paths_dir_names_for_inference = ['1_12','12_24','24_36','36_60'] # for instance for alot of models we want to inference: ['model_1','model_2' ... ] for one : ['model_1']

###########################################################################################################################################################
base_path = local_path
output_path = os.path.join(base_path, name_of_the_model_to_save_train)
checkpoints_path = os.path.join(output_path, 'checkpoints')
scalers_path = os.path.join(output_path, 'scalers')
inference_output_path = os.path.join(output_path, 'inference_output')

STATIONS_COORDINATES = {
    'Tavor Kadoorie':           (238440, 734540), #station id: 13
    'Newe Yaar':                (217010, 734820), #station id: 186
    'Yavneel':                  (248110, 733730), #station id: 11
    'En Hashofet':              (209310, 723170), #station id: 67
    'Eden Farm':                (246190, 708240), #station id: 206
    'Eshhar':                   (228530, 754390), #station id: 205
    'Afula Nir Haemeq':         (226260, 722410)  #station id: 16
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
    'input_width' :     72, # window input size
    'label_width' :     12, # how many hours to predict to the future
    'shift' :           1,
    'label_columns' :   ['TD (degC)'],
}

"""
WINDOW_PARAMS = {
    'input_width' :     72, # window input size
    'label_width' :     12, # how many hours to predict to the future
    'shift' :           1,
    'label_columns' :   ['TD (degC)'],
}
"""
"""
WINDOW_PARAMS = {
    'input_width' :     72, # window input size
    'label_width' :     12, # how many hours to predict to the future
    'shift' :           13,
    'label_columns' :   ['TD (degC)'],
}
"""
"""
WINDOW_PARAMS = {
    'input_width' :     72, # window input size
    'label_width' :     12, # how many hours to predict to the future
    'shift' :           25,
    'label_columns' :   ['TD (degC)'],
}
"""
"""
WINDOW_PARAMS = {
    'input_width' :     72, # window input size
    'label_width' :     24, # how many hours to predict to the future
    'shift' :           37,
    'label_columns' :   ['TD (degC)'],
}
"""

TRAIN_PARAMS = {
    'epochs' :          50,
    'batch_size':       32,
    'lr':               1e-5,                                   # 1e-3, 1e-4, 1e-5
    'checkpoint_dir' :  PARAMS['checkpoints_path'],
    'resume':           False,
    'device':           PARAMS['device'],
    'early_stopping_patience':10,                               # how many epochs to wait before stopping the training
    'scheduler_patience':3,                                     # how many epochs to wait before reducing the learning rate
    'scheduler_factor':  0.5,                                   # the factor to reduce the learning rate
    'min_lr':            1e-7,
    'logger_path':       PARAMS['output_path']
}

ADVANCED_MODEL_PARAMS = {
    'num_stations':         len(PARAMS['fileNames']),
    'time_steps':           WINDOW_PARAMS['input_width'],
    'feature_dim':          PARAMS['in_channels'],
    'kernel_size':          3,  # cnn filter size                       4, 5, 6, 7
    'd_model':              64, # input for transformer size            64, 128
    'nhead':                8,  # number of heads in the transformer    8, 16
    'num_layers':           4,  # number of layers in the transformer - 6 - 12
    'target_station_idx':   PARAMS['target_station_id'],
    'label_width':          WINDOW_PARAMS['label_width'],
    'output_per_feature':   3,                                          # 4 ,5
    'use_batch_norm':       False,
    'use_residual':         True
}

models_paths_dir_names_full_paths = [os.path.join(base_path, model_folder_name) for model_folder_name in models_paths_dir_names_for_inference]

INFERENCE_PARAMS = {
    'params_path':             [os.path.join(folder, 'parameters.py') for folder in models_paths_dir_names_full_paths],
    'weights_paths':           [os.path.join(folder, 'checkpoints', 'best_checkpoint.pth') for folder in models_paths_dir_names_full_paths],
    'scaler_folder_path':      PARAMS['scalers_path'],
    'inference_output_path_per_model':  models_paths_dir_names_full_paths, # for saving the output of the inference in the model folder for each model
    'inference_output_path':  os.path.join(base_path, 'inference_output'), # for saving the output of the inference of all models in one folder (later analyze.py will use it)
}