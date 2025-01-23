# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from window_generator import WindowGenerator
# Instead of importing CNNTransformerModel, we'll import LSTMModel
from model import LSTMModel
from backend.Model_Pytorch.data import loadData

def train_model(
    train_dataset,
    val_dataset,
    model,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    checkpoint_dir='./checkpoints',
    resume=False,
    device='cpu'
):
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Check if GPU is available and set device accordingly

    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    # Resume training if needed
    if resume and os.path.exists(latest_ckpt):
        print("Resuming training from latest checkpoint...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)  # (batch, seq_len, in_channels)
            y_batch = y_batch.to(device)  # (batch, label_width, num_label_features) or (batch, 1, 1)
            
            optimizer.zero_grad()
            preds = model(x_batch)  # (batch, 1)
            
            # Flatten y_batch if necessary
            y_batch_single = y_batch.squeeze(-1).squeeze(-1)  # (batch,)
            preds = preds.squeeze(-1)                         # (batch,)
            
            loss = criterion(preds, y_batch_single)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                preds_val = model(x_val).squeeze(-1)   # (batch,)
                y_val_single = y_val.squeeze(-1).squeeze(-1)  # (batch,)
                
                loss_val = criterion(preds_val, y_val_single)
                val_losses.append(loss_val.item())
        
        val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Checkpoint: always save the 'latest'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, latest_ckpt)
        
        # If best val, save a special "best" checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_ckpt)
            print(f"  -> Best model saved at epoch {epoch+1} (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    # Example usage:
    
    # 1) Load your single-station data
    df = loadData("Afeq")[0]
    
    # Convert to numpy array
    data_np = df.values  # shape (T, in_channels)
    
    # 2) Create train/val split
    train_size = int(0.8 * len(data_np))
    train_data = data_np[:train_size]
    val_data   = data_np[train_size:]
    
    # 3) Create Datasets
    input_width = 24
    label_width = 1
    shift = 1
    label_columns = [-1]  
    train_dataset = WindowGenerator(train_data, input_width, label_width, shift, label_columns)
    val_dataset   = WindowGenerator(val_data,   input_width, label_width, shift, label_columns)
    
    # 4) Instantiate model (LSTM)
    in_channels = df.shape[1]  # number of features
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define your LSTM model with desired parameters
    model = LSTMModel(
        in_channels=in_channels,
        hidden_dim=64,
        num_layers=2,
        label_width=label_width
    )
    
    # 5) Train
    train_model(
        train_dataset,
        val_dataset,
        model,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        checkpoint_dir='./checkpoints',
        resume=False,
        device=device
    )
