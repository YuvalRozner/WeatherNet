# inference.py

import torch
import numpy as np
import pandas as pd

from model import LSTMModel
from backend.Model_Pytorch.common.data import preprocessing_tensor_df
import pickle
import os
from parameters import PARAMS, WINDOW_PARAMS, LSTM_MODEL_PARAMS
from tqdm import tqdm
import matplotlib.pyplot as plt
from backend.Model_Pytorch.common.data import preprocessing_our_df, normalize_data, load_pkl_file

from backend.Model_Pytorch.common.window_generator import WindowGenerator


# inference.py

def load_scaler(scaler_path='./scaler.pkl'):
    """
    Load the previously saved scaler.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")
    return scaler


def load_model_for_inference(checkpoint_path, model_params, device='cpu'):
    """
    Create a model with the same architecture,
    load checkpoint, and return it in eval mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model = LSTMModel(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def load_data_and_preprocess(data_path, target_column):
    df = load_pkl_file(PARAMS['fileName'])
    df = preprocessing_our_df(df)
    # df = pd.read_csv(data_path)
    # df = preprocessing_tensor_df(df)
    target_index = df.columns.get_loc(target_column)
    return df, target_index


def load_window(data_np, window_size, shift, label_width, scaler, target_column_index=1, idx=0):
    # Calculate the number of possible windows
    total_windows = len(data_np) - window_size - 1  # -1 for target
    if total_windows < 1:
        raise ValueError(f"Not enough data points ({len(data_np)}) for window size {window_size} and target.")

    # Extract the window and actual target
    window = data_np[idx:idx + window_size, :]  # shape (window_size, in_channels)
    actual_target = data_np[idx + window_size + shift - 1:idx + window_size + label_width + shift - 1,
                    target_column_index]  # shape (1,)
    # Normalize the window
    window_scaled = scaler.transform(window)
    # Convert to torch.Tensor and reshape to (1, window_size, in_channels)
    window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)

    return window_tensor, actual_target


@torch.no_grad()
def predict(model, input_window, target_column_index, device='cpu'):
    """
    Perform prediction on a single input window.

    Args:
        model (torch.nn.Module): Trained LSTM model.
        input_window (torch.Tensor): Input data of shape (1, seq_len, in_channels).
        device (str): Device to perform inference on.

    Returns:
        float: Predicted temperature in original scale (°C).
    """
    input_tensor = input_window.to(device)

    # Model prediction (scaled)
    output_scaled = model(input_tensor)  # shape (batch, 1)

    # Convert to numpy
    output_scaled_np = output_scaled.squeeze(-1).cpu().numpy().reshape(-1, 1)  # Shape: (batch_size, 1)

    dummy = np.zeros((output_scaled_np.shape[0], scaler.mean_.shape[0]))
    dummy[:, target_column_index] = output_scaled_np[:, 0]

    # Inverse transform
    output_original_scale = scaler.inverse_transform(dummy)[:, target_column_index]

    return output_original_scale  # Return as scalar


def get_val_data(df):
    data_np = df.values  # shape (T, in_channels)
    train_size = int(0.8 * len(data_np))
    val_data = data_np[train_size:]
    return val_data


if __name__ == "__main__":
    # Define parameters directly in main
    path_to_file = os.path.join(os.path.dirname(__file__), PARAMS['filePah'])

    data_path = path_to_file  # Adjust the path as needed
    scaler_path = os.path.join(os.path.dirname(__file__), 'output', 'scaler.pkl')
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'output', 'checkpoints', 'best_checkpoint.pth')
    window_size = WINDOW_PARAMS['input_width']  # Must match input_width used during training
    target_column = WINDOW_PARAMS['label_columns'][0]  # Ensure this matches your dataset
    shift = WINDOW_PARAMS['shift']
    prediction_mode = 'train'  # Options: 'single', 'train'
    ## single parameters
    index = 500  # index for single

    ## train parameters
    stop_after = 30  # if you dont want to stop set to -1
    show_examples = 30  # how many examples to plot
    plot = True
    if show_examples > stop_after:
        raise RuntimeError("show_examples needs to be lower then stop_after")

    # Define model parameters
    model_params = {
        "in_channels": 15,  # Must match your data's feature count
        "hidden_dim": LSTM_MODEL_PARAMS['hidden_dim'],
        "num_layers": LSTM_MODEL_PARAMS['num_layers'],
        "label_width": WINDOW_PARAMS['label_width'],
    }
    df, target_index = load_data_and_preprocess(data_path, target_column)

    device = PARAMS['device']

    model = load_model_for_inference(checkpoint_path, model_params, device=device)  # Load model

    scaler = load_scaler(scaler_path=scaler_path)  # Load scaler

    predictions = []
    actual_temps = []

    if prediction_mode == 'single':
        scaled_window, actual_temp = load_window(data_np=df.values, window_size=window_size, shift=shift,
                                                 label_width=model_params['label_width'],
                                                 scaler=scaler, target_column_index=target_index, idx=index)
        y_pred = predict(model, scaled_window, target_index, device=device)
        print(f"Predicted temperature (°C): {y_pred}")
        print(f"Actual temperature (°C): {actual_temp}")
    elif prediction_mode == 'train':
        # for each of val_data_scaled check prediction against actual
        val_data = get_val_data(df)

        end = len(val_data) - window_size if stop_after == -1 else min(len(val_data) - window_size,
                                                                       stop_after)
        for i in tqdm(range(0, end, model_params['label_width']), desc="Predicting"):
            scaled_window, actual_temp = load_window(data_np=val_data, window_size=window_size, shift=shift,
                                                     label_width=model_params['label_width'],
                                                     scaler=scaler, target_column_index=target_index, idx=i)
            y_pred = predict(model, scaled_window, target_index, device=device)
            predictions.append(y_pred)
            actual_temps.append(actual_temp)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(range(30), actual_temps[:30], label='Actual', color='blue')
            plt.scatter(range(30), predictions[:30], label='Predicted', color='red')
            plt.xlabel('Hours')
            plt.ylabel('Temperature (°C)')
            plt.title('Temperature Prediction for the next week')
            plt.legend()
            plt.show()

    else:
        print(f"Invalid prediction mode: {prediction_mode}.")
