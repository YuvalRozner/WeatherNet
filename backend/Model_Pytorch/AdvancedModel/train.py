import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def train_model(
    train_dataset,
    val_dataset,
    model,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    checkpoint_dir='./checkpoints',
    resume=False,
    device='cpu',
    coord=[0, 0]
):
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs}")):
            x_batch = x_batch.to(device)  # [batch_size, input_width, num_stations, num_features]
            y_batch = y_batch.to(device)  # [batch_size, num_label_features]
            
            # Forward pass with coordinates
            preds = model(x_batch, coord[0].to(device), coord[1].to(device))  # [batch_size, 1]
            
            # Flatten y_batch if necessary
            y_batch_single = y_batch.squeeze(-1)  # [batch_size]
            preds = preds.squeeze(-1)            # [batch_size]
            
            # Compute loss
            loss = criterion(preds, y_batch_single)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Compute average loss for the epoch
        train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                # Forward pass with coordinates
                preds_val = model(x_val, coord[0].to(device), coord[1].to(device))  # [batch_size, 1]
                
                # Flatten predictions and labels
                preds_val = preds_val.squeeze(-1)          # [batch_size]
                y_val_single = y_val.squeeze(-1)          # [batch_size]
                
                # Compute loss
                loss_val = criterion(preds_val, y_val_single)
                val_losses.append(loss_val.item())
        
        # Compute average validation loss
        val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}/{start_epoch + epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
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
