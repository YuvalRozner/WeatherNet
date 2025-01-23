# inference.py

import torch
import numpy as np

from model import CNNTransformerModel

def load_model_for_inference(checkpoint_path, model_params, device='cpu'):
    """
    Create a model with the same architecture, 
    load checkpoint, and return it in eval mode.
    
    model_params: dict with the same hyperparams used to init CNNTransformerModel
    """
    model = CNNTransformerModel(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(model, input_window, device='cpu'):
    """
    input_window: torch.Tensor, shape (1, seq_len, in_channels)
        or (batch_size, seq_len, in_channels)
    Returns:
        prediction: (batch_size,) or (batch_size, 1)
    """
    input_window = input_window.to(device)
    output = model(input_window)  # shape (batch, 1)
    return output.squeeze(-1).cpu().numpy()

if __name__ == "__main__":
    # Example usage:
    # 1) define same model_params as in training
    model_params = {
        "in_channels": 8,       # must match your data
        "d_model": 64,
        "nhead": 4,
        "num_transformer_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "cnn_kernel_size": 3,
        "label_width": 1,
    }
    
    # 2) load model
    best_ckpt = "./checkpoints/best_checkpoint.pth"
    model = load_model_for_inference(best_ckpt, model_params, device='cpu')
    
    # 3) get your "last window" (shape (1, seq_len, in_channels))
    last_window_np = np.random.randn(1, 24, 8).astype(np.float32)  # Example
    last_window_tensor = torch.from_numpy(last_window_np)
    
    # 4) predict
    y_pred = predict(model, last_window_tensor)
    print("Predicted temperature:", y_pred)
