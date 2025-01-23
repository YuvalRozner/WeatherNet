# inference.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from model import LSTMModel
from window_generator import WindowGenerator
from sklearn.preprocessing import StandardScaler
import pickle
import os
import csv

# inference.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from model import LSTMModel
from window_generator import WindowGenerator
from sklearn.preprocessing import StandardScaler
import pickle
import os
import csv


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


def preprocessing_df(df):
    """
    Apply the same preprocessing steps as during training.
    """
    # Slice the DataFrame and create a copy to avoid SettingWithCopyWarning
    df = df[5::6].copy()
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # Handle 'wv (m/s)'
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    df.loc[bad_wv, 'wv (m/s)'] = 0.0  # Use .loc to modify the original DataFrame
    wv = df.pop('wv (m/s)')

    # Handle 'max. wv (m/s)'
    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    df.loc[bad_max_wv, 'max. wv (m/s)'] = 0.0  # Use .loc to modify the original DataFrame
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate wind x and y components using .loc
    df.loc[:, 'Wx'] = wv * np.cos(wd_rad)
    df.loc[:, 'Wy'] = wv * np.sin(wd_rad)
    df.loc[:, 'max Wx'] = max_wv * np.cos(wd_rad)
    df.loc[:, 'max Wy'] = max_wv * np.sin(wd_rad)

    # Time-based features
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day

    df.loc[:, 'Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df.loc[:, 'Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df.loc[:, 'Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df.loc[:, 'Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


def load_random_window(data_path, window_size, scaler, target_column='T (degC)', seed=42):
    """
    Load data, preprocess, normalize, and extract a random window and its actual target for inference.

    Args:
        data_path (str): Path to the CSV data file.
        window_size (int): Number of time steps in the input window.
        scaler (StandardScaler): Fitted scaler for data normalization.
        target_column (str): Name of the target column.
        seed (int): Seed for random number generator.

    Returns:
        torch.Tensor: Random window of shape (1, window_size, in_channels).
        float: Actual temperature corresponding to the window.
    """
    import random

    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}")

    # Preprocess data
    df = preprocessing_df(df)
    print("Data preprocessing completed.")

    # Convert to numpy array
    data_np = df.values  # shape (T, in_channels)
    print(f"Data shape after preprocessing: {data_np.shape}")

    # Find the index of the target column
    try:
        target_index = df.columns.get_loc(target_column)
    except KeyError:
        raise KeyError(f"Target column '{target_column}' not found in the data.")

    # Calculate the number of possible windows
    total_windows = len(data_np) - window_size - 1  # -1 for target
    if total_windows < 1:
        raise ValueError(f"Not enough data points ({len(data_np)}) for window size {window_size} and target.")

    # Select a random window index
    random_idx = random.randint(0, total_windows - 1)
    print(f"Selected random window index: {random_idx}")

    # Extract the window and actual target
    window = data_np[random_idx:random_idx + window_size, :]  # shape (window_size, in_channels)
    actual_target = data_np[random_idx + window_size, target_index]  # shape (1,)

    print(f"Random window shape: {window.shape}")
    print(f"Actual target temperature: {actual_target}")

    # Normalize the window
    window_scaled = scaler.transform(window)
    print("Random window normalized.")

    # Convert to torch.Tensor and reshape to (1, window_size, in_channels)
    window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
    print(f"Random window reshaped for model input: {window_tensor.shape}")

    return window_tensor, actual_target


def load_last_window(data_path, window_size, scaler, target_column='T (degC)'):
    """
    Load data, preprocess, normalize, and extract the last window and its actual target for inference.

    Args:
        data_path (str): Path to the CSV data file.
        window_size (int): Number of time steps in the input window.
        scaler (StandardScaler): Fitted scaler for data normalization.
        target_column (str): Name of the target column.

    Returns:
        torch.Tensor: Last window of shape (1, window_size, in_channels).
        float: Actual temperature corresponding to the window.
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}")

    # Preprocess data
    df = preprocessing_df(df)
    print("Data preprocessing completed.")

    # Convert to numpy array
    data_np = df.values  # shape (T, in_channels)
    print(f"Data shape after preprocessing: {data_np.shape}")

    # Find the index of the target column
    try:
        target_index = df.columns.get_loc(target_column)
    except KeyError:
        raise KeyError(f"Target column '{target_column}' not found in the data.")

    # Check if there are enough data points for window and target
    if len(data_np) < window_size + 1:
        raise ValueError(f"Not enough data points ({len(data_np)}) for window size {window_size} and target.")

    # Extract the last window and the actual target
    last_window = data_np[-(window_size + 1):-1, :]  # shape (window_size, in_channels)
    actual_target = data_np[-1, target_index]  # shape (1,)

    print(f"Last window shape: {last_window.shape}")
    print(f"Actual target temperature: {actual_target}")

    # Normalize the last window
    last_window_scaled = scaler.transform(last_window)
    print("Last window normalized.")

    # Convert to torch.Tensor and reshape to (1, window_size, in_channels)
    last_window_tensor = torch.tensor(last_window_scaled, dtype=torch.float32).unsqueeze(0)
    print(f"Last window reshaped for model input: {last_window_tensor.shape}")

    return last_window_tensor, actual_target


def load_windows(data_path, window_size, scaler, target_column='T (degC)'):
    """
    Load data, preprocess, normalize, and extract all windows and their actual targets for inference.

    Args:
        data_path (str): Path to the CSV data file.
        window_size (int): Number of time steps in the input window.
        scaler (StandardScaler): Fitted scaler for data normalization.
        target_column (str): Name of the target column.

    Returns:
        list of torch.Tensor: List of windows, each of shape (1, window_size, in_channels).
        list of float: List of actual temperatures corresponding to each window.
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}")

    # Preprocess data
    df = preprocessing_df(df)
    print("Data preprocessing completed.")

    # Convert to numpy array
    data_np = df.values  # shape (T, in_channels)
    print(f"Data shape after preprocessing: {data_np.shape}")

    # Find the index of the target column
    try:
        target_index = df.columns.get_loc(target_column)
    except KeyError:
        raise KeyError(f"Target column '{target_column}' not found in the data.")

    # Check if there are enough data points for at least one window and target
    if len(data_np) < window_size + 1:
        raise ValueError(f"Not enough data points ({len(data_np)}) for window size {window_size} and target.")

    windows = []
    actuals = []

    for i in range(len(data_np) - window_size):
        window = data_np[i:i + window_size, :]
        target = data_np[i + window_size, target_index]

        # Normalize the window
        window_scaled = scaler.transform(window)
        window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(
            0)  # Shape: (1, window_size, in_channels)

        windows.append(window_tensor)
        actuals.append(target)

    print(f"Total windows loaded: {len(windows)}")

    return windows, actuals


@torch.no_grad()
def predict_batch(model, windows, device='cpu'):
    """
    Perform batch predictions.

    Args:
        model (torch.nn.Module): Trained LSTM model.
        windows (list of torch.Tensor): List of input windows, each of shape (1, seq_len, in_channels).
        device (str): Device to perform inference on.

    Returns:
        list of float: Predicted temperatures in original scale (°C).
    """
    predictions = []
    for window in windows:
        pred = predict(model, window, device)
        predictions.append(pred)
    return predictions


@torch.no_grad()
def predict(model, input_window, device='cpu'):
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

    # Inverse transform to get original scale
    # Create a dummy array with the same number of features as the scaler expects
    # Only replace the target column with the scaled prediction
    # Assuming 'T (degC)' is the last column
    dummy = np.zeros((output_scaled_np.shape[0], scaler.mean_.shape[0]))
    dummy[:, -1] = output_scaled_np[:, 0]

    # Inverse transform
    output_original_scale = scaler.inverse_transform(dummy)[:, -1]

    return output_original_scale[0]  # Return as scalar


def load_random_window(data_path, window_size, scaler, target_column='T (degC)', seed=42):
    """
    Load data, preprocess, normalize, and extract a random window and its actual target for inference.

    Args:
        data_path (str): Path to the CSV data file.
        window_size (int): Number of time steps in the input window.
        scaler (StandardScaler): Fitted scaler for data normalization.
        target_column (str): Name of the target column.
        seed (int): Seed for random number generator.

    Returns:
        torch.Tensor: Random window of shape (1, window_size, in_channels).
        float: Actual temperature corresponding to the window.
    """
    import random

    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}")

    # Preprocess data
    df = preprocessing_df(df)
    print("Data preprocessing completed.")

    # Convert to numpy array
    data_np = df.values  # shape (T, in_channels)
    print(f"Data shape after preprocessing: {data_np.shape}")

    # Find the index of the target column
    try:
        target_index = df.columns.get_loc(target_column)
    except KeyError:
        raise KeyError(f"Target column '{target_column}' not found in the data.")

    # Calculate the number of possible windows
    total_windows = len(data_np) - window_size - 1  # -1 for target
    if total_windows < 1:
        raise ValueError(f"Not enough data points ({len(data_np)}) for window size {window_size} and target.")

    # Select a random window index
    random_idx = random.randint(0, total_windows - 1)
    print(f"Selected random window index: {random_idx}")

    # Extract the window and actual target
    window = data_np[random_idx:random_idx + window_size, :]  # shape (window_size, in_channels)
    actual_target = data_np[random_idx + window_size, target_index]  # shape (1,)

    print(f"Random window shape: {window.shape}")
    print(f"Actual target temperature: {actual_target}")

    # Normalize the window
    window_scaled = scaler.transform(window)
    print("Random window normalized.")

    # Convert to torch.Tensor and reshape to (1, window_size, in_channels)
    window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
    print(f"Random window reshaped for model input: {window_tensor.shape}")

    return window_tensor, actual_target


def save_predictions(predictions, actuals, output_path='predictions.csv'):
    """
    Save predictions and actual temperatures to a CSV file.

    Args:
        predictions (list of float): Predicted temperatures.
        actuals (list of float): Actual temperatures.
        output_path (str): Path to save the CSV file.
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prediction (°C)', 'Actual (°C)'])
        for pred, actual in zip(predictions, actuals):
            writer.writerow([f"{pred:.2f}", f"{actual:.2f}"])
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Define parameters directly in main
    data_path = "../utils/jena_climate_2009_2016.csv"  # Adjust the path as needed
    scaler_path = './scaler.pkl'
    checkpoint_path = "./checkpoints/best_checkpoint.pth"
    window_size = 24  # Must match input_width used during training
    target_column = 'T (degC)'  # Ensure this matches your dataset
    output_csv = 'predictions.csv'  # Path to save predictions
    seed = 80  # Seed for random window selection

    # Define model parameters
    model_params = {
        "in_channels": 19,  # Must match your data's feature count
        "hidden_dim": 64,
        "num_layers": 2,
        "label_width": 1,
    }

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model_for_inference(checkpoint_path, model_params, device=device)

    # Load scaler
    scaler = load_scaler(scaler_path=scaler_path)

    # Choose prediction mode: 'single', 'random', or 'batch'
    prediction_mode = 'random'  # Options: 'single', 'random', 'batch'

    if prediction_mode == 'single':
        # Load the last window and actual temperature for single prediction
        last_window_scaled, actual_temp = load_last_window(data_path, window_size, scaler, target_column)

        # Make single prediction
        y_pred = predict(model, last_window_scaled, device=device)

        # Print both predicted and actual temperatures
        print(f"Predicted temperature (°C): {y_pred:.2f}")
        print(f"Actual temperature (°C): {actual_temp:.2f}")

    elif prediction_mode == 'random':
        # Load a random window and its actual temperature
        random_window_scaled, actual_temp = load_random_window(data_path, window_size, scaler, target_column, seed=seed)

        # Make prediction on the random window
        y_pred = predict(model, random_window_scaled, device=device)

        # Print both predicted and actual temperatures
        print(f"Random Window Prediction:")
        print(f"Predicted temperature (°C): {y_pred:.2f}")
        print(f"Actual temperature (°C): {actual_temp:.2f}")

        random_window_scaled, actual_temp = load_random_window(data_path, window_size, scaler, target_column, seed=seed+1)

        # Make prediction on the random window
        y_pred = predict(model, random_window_scaled, device=device)

        # Print both predicted and actual temperatures
        print(f"Random Window Prediction:")
        print(f"Predicted temperature (°C): {y_pred:.2f}")
        print(f"Actual temperature (°C): {actual_temp:.2f}")

        random_window_scaled, actual_temp = load_random_window(data_path, window_size, scaler, target_column,
                                                               seed=seed + 2)

        # Make prediction on the random window
        y_pred = predict(model, random_window_scaled, device=device)

        # Print both predicted and actual temperatures
        print(f"Random Window Prediction:")
        print(f"Predicted temperature (°C): {y_pred:.2f}")
        print(f"Actual temperature (°C): {actual_temp:.2f}")

    elif prediction_mode == 'batch':
        # Load all windows and actual temperatures
        windows_scaled, actual_temps = load_windows(data_path, window_size, scaler, target_column)

        # Make batch predictions
        predictions = predict_batch(model, windows_scaled, device=device)

        # Save predictions to CSV
        save_predictions(predictions, actual_temps, output_path=output_csv)

        # Calculate and display performance metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(actual_temps, predictions)
        rmse = np.sqrt(mean_squared_error(actual_temps, predictions))

        print(f"\nOverall Performance:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}°C")

    else:
        print(f"Invalid prediction mode: {prediction_mode}. Choose from 'single', 'random', or 'batch'.")
