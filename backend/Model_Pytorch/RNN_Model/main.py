# main.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from window_generator import WindowGenerator

from model import LSTMModel
from train import train_model

from sklearn.preprocessing import StandardScaler
import pickle

def preproccecingDf(df):
    df = df[5::6]
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    
    # Handle 'wv (m/s)'
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0
    wv = df.pop('wv (m/s)')
    
    # Handle 'max. wv (m/s)'
    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0
    max_wv = df.pop('max. wv (m/s)')
    
    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180
    
    # Calculate wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)
    
    # Time-based features
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    
    return df

def normalize_data(train_data, val_data, scaler_path='./scaler.pkl'):
    """
    Fit a StandardScaler on the training data and transform both train and val data.
    Save the scaler to disk for future use.
    
    Args:
        train_data (np.ndarray): Training data.
        val_data (np.ndarray): Validation data.
        scaler_path (str): Path to save the scaler.
        
    Returns:
        train_data_scaled (np.ndarray): Scaled training data.
        val_data_scaled (np.ndarray): Scaled validation data.
        scaler (StandardScaler): Fitted scaler object.
    """
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    train_data_scaled = scaler.transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    
    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaler saved to {scaler_path}")
    
    return train_data_scaled, val_data_scaled, scaler

if __name__ == "__main__":
    # 1) Load your single-station data
    df = pd.read_csv("..\\utils\\jena_climate_2009_2016.csv")
    df = preproccecingDf(df)
    
    # 2) Convert to numpy array
    data_np = df.values  # shape (T, in_channels)
    
    # 3) Create train/val split
    train_size = int(0.8 * len(data_np))
    train_data = data_np[:train_size]
    val_data = data_np[train_size:]
    
    # 4) Normalize the data
    train_data_scaled, val_data_scaled, scaler = normalize_data(train_data, val_data, scaler_path='./scaler.pkl')
    
    # 5) Create Datasets
    input_width = 24
    label_width = 1
    shift = 1
    column_indices = {name: i for i, name in enumerate(df.columns)}
    label_columns = [column_indices['T (degC)']]
    
    train_dataset = WindowGenerator(train_data_scaled, input_width, label_width, shift, label_columns)
    val_dataset   = WindowGenerator(val_data_scaled,   input_width, label_width, shift, label_columns)
    
    # 6) Instantiate model (LSTM)
    in_channels = df.shape[1]  # number of features
    print(f"Number of features: {in_channels}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Define your LSTM model with desired parameters
    model = LSTMModel(
        in_channels=in_channels,
        hidden_dim=64,
        num_layers=2,
        label_width=label_width
    )
    epochs_ = 2
    # 7) Train
    train_model(
        train_dataset,
        val_dataset,
        model,
        epochs=epochs_,
        batch_size=32,
        lr=1e-3,
        checkpoint_dir='./checkpoints',
        resume=False,
        device=device
    )
