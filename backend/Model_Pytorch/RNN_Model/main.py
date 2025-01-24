# main.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from window_generator import WindowGenerator

from model import LSTMModel
from train import train_model
from common.data import preprocessing_tensor_df, normalize_data

if __name__ == "__main__":

    df = pd.read_csv("..\\common\\jena_climate_2009_2016.csv")
    df = preprocessing_tensor_df(df)

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
    print(f"column_indices['T (degC)']: {column_indices['T (degC)']}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Define your LSTM model with desired parameters

    model = LSTMModel(
        in_channels=in_channels,
        hidden_dim=64,
        num_layers=2,
        label_width=label_width
    )

    epochs_ = 20
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
