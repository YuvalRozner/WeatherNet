import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

class TimeSeriesWindowDataset(Dataset):
    """
    Creates a sliding window dataset from a 2D array [time, features].
    
    Args:
        data (numpy.ndarray or torch.Tensor): 
            Shape = [total_time, num_features].
        input_width (int): 
            Number of timesteps of input.
        label_width (int): 
            Number of timesteps of output labels.
        shift (int): 
            How far ahead the label is from the start of the input window.
        label_columns (list of int or list of str, optional): 
            Which columns in `data` are the labels? 
            If None, all features are used as labels.
        column_indices (dict, optional): 
            A mapping from column names -> index in data. 
            Only needed if label_columns is a list of strings.
    """
    def __init__(
        self, 
        data,
        input_width,
        label_width,
        shift,
        label_columns=None,
        column_indices=None
    ):
        super().__init__()
        
        # Make sure `data` is a torch.Tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        self.data = data
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        
        # If we have label columns by name, convert them to indices
        if label_columns is not None and column_indices is not None:
            # e.g., label_columns = ["TD (degC)"] -> map to an int index
            self.label_columns_indices = [column_indices[name] for name in label_columns]
        elif label_columns is not None and column_indices is None:
            # If label_columns are already indices, just store them
            self.label_columns_indices = label_columns
        else:
            # All columns are used as labels
            self.label_columns_indices = None
        
        # Number of samples is how many windows fit in the time dimension
        # e.g. if `len(data) = 1000`, `total_window_size = 24`, then we have 1000-24+1 windows
        self.num_samples = len(self.data) - self.total_window_size + 1
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # The window starts at index `idx` and ends at `idx + total_window_size`
        x_start = idx
        x_end = idx + self.input_width
        y_start = x_end + (self.shift - self.input_width)
        y_end = y_start + self.label_width
        
        # Inputs: [input_width, num_features]
        x = self.data[x_start:x_end]
        
        # Labels: [label_width, num_label_features]
        if self.label_columns_indices is not None:
            y = self.data[y_start:y_end, self.label_columns_indices]
        else:
            y = self.data[y_start:y_end]
        
        return x, y

if __name__ == "__main__":


    # Let's assume df is your entire dataset, e.g. with shape [time, features].
    df = pd.read_csv("my_timeseries.csv")  # Example
    data_array = df.values  # shape = (time, features)

    # If you want to forecast the "TD (degC)" column:
    label_columns = ["TD (degC)"]
    column_indices = {name: i for i, name in enumerate(df.columns)}  # mapping col->index

    # Create the dataset:
    input_width = 24  # e.g. 24 hours as input
    label_width = 1   # e.g. forecast 1 hour ahead
    shift = 1         # how far ahead

    train_dataset = TimeSeriesWindowDataset(
        data=data_array,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        label_columns=label_columns,
        column_indices=column_indices
    )

    # Create a Dataloader:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop snippet:
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            # x_batch shape = [batch_size, input_width, num_features]
            # y_batch shape = [batch_size, label_width, num_label_features]
            
            # Move to device if using GPU:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass:
            predictions = model(x_batch)
            
            # MSE loss example (if label_width=1, you might flatten or adapt accordingly)
            loss = criterion(predictions, y_batch.squeeze(-1))  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
