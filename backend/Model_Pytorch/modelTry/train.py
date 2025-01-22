import os
import torch
import numpy as np
from model import TemperatureForecastModel 

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    checkpoint_dir='./checkpoints',
    resume=False
):
    """
    Train a PyTorch model with checkpointing.
    
    Args:
        model (nn.Module): Your PyTorch model.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        criterion: Loss function (e.g. nn.MSELoss()).
        optimizer: Optimizer (e.g. torch.optim.Adam(model.parameters(), ...)).
        device: 'cpu' or 'cuda'.
        epochs (int): Number of epochs to train.
        checkpoint_dir (str): Directory to save checkpoints.
        resume (bool): If True, attempt to resume from 'latest_checkpoint.pth'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0
    
    # If resume=True, load from the latest checkpoint
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if resume and os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed training from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    model.to(device)

    for epoch in range(start_epoch, epochs):
        # ------------------------- TRAIN -------------------------
        model.train()
        train_losses = []
        for (x_batch, y_batch) in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            
            # Reshape if needed, depending on your model's output shape
            # e.g., preds: (batch_size, 1), y_batch: (batch_size, 1)
            loss = criterion(preds, y_batch)
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)

        # ------------------------- VALIDATION -------------------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (x_val, y_val) in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                preds_val = model(x_val)
                loss_val = criterion(preds_val, y_val)
                val_losses.append(loss_val.item())
        val_loss = np.mean(val_losses)

        # Print training progress
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ------------------------- CHECKPOINTING -------------------------
        # Save "latest" checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, latest_checkpoint_path)

        # If this is the best validation so far, save a "best" checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch+1} with val_loss={val_loss:.4f}")

def load_model_for_inference(model, checkpoint_path, device='cpu'):
    """
    Load the model's weights from a checkpoint for inference.
    
    Args:
        model (nn.Module): A *freshly created* instance of the same architecture.
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        nn.Module: The model loaded with checkpoint weights, in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer(model, input_window, device='cpu'):
    """
    Run inference on the given input.
    
    Args:
        model (nn.Module): The model loaded with pretrained weights.
        input_window (torch.Tensor): 
            The window of input data, shape [batch_size, ...] 
            (depending on your model's expected shape).
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        torch.Tensor: Model's prediction(s).
    """
    input_window = input_window.to(device)
    predictions = model(input_window)
    return predictions


if __name__ == "__main__":

    # Suppose you have a last window of shape (1, num_stations, in_channels, seq_len)
    # or (1, in_channels, seq_len) if single station. Adjust to your model's input shape.
    
    # 1) Recreate the model
    model_for_inference = TemperatureForecastModel(
        num_stations=5,
        target_station_idx=0,
        in_channels_per_station=10,
        seq_len=24,
        cnn_embed_dim=64,
        loc_emb_dim=32,
        d_model=128,
        nhead=4,
        num_transformer_layers=2
    )

    # 2) Load checkpoint
    best_checkpoint_path = "./checkpoints/best_checkpoint.pth"
    model_for_inference = load_model_for_inference(model_for_inference, best_checkpoint_path, device='cpu')

    # 3) Prepare input (e.g. last window)
    #    In real code, you'd gather the final 24 hours (or however many) for each station.
    last_window = torch.randn(1, 5, 10, 24)  # shape depends on your model

    # 4) Location embeddings if your model also requires station coords
    #    If your model expects both timeseries + locations, adapt the code accordingly.
    #    Example:
    station_locations = torch.randn(1, 5, 2)

    # 5) Inference
    with torch.no_grad():
        # If your model forward signature is: forward(timeseries, locations)
        prediction = model_for_inference(last_window, station_locations)
        print("Prediction shape:", prediction.shape)
        print("Prediction:", prediction)
