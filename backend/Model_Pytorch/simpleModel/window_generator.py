# window_generator.py

import torch
from torch.utils.data import Dataset
import numpy as np

class WindowGenerator(Dataset):
    """
    Creates sliding windows from a single-station dataset.
    data: shape (T, num_features)
    input_width (int): window size for input
    label_width (int): how many time steps to predict
    shift (int): how far ahead the prediction starts after the input
    label_columns (list[int] or None): indices of the columns used as labels
    """
    def __init__(
        self,
        data,
        input_width,
        label_width,
        shift,
        label_columns=None,
    ):
        super().__init__()
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        self.data = data  # shape (T, num_features)
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        # total window = input plus how far to shift
        self.total_window_size = input_width + shift
        
        # label columns
        self.label_columns = label_columns
        # If None, we'll just return all features as labels
        if label_columns is not None:
            self.num_label_features = len(label_columns)
        else:
            self.num_label_features = self.data.shape[-1]
        
        # how many total samples (start points) we can have
        self.num_samples = len(self.data) - self.total_window_size - (label_width - 1)
        
        if self.num_samples < 1:
            raise ValueError("Not enough data to create windows with these parameters.")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # input window range
        x_start = idx
        x_end = x_start + self.input_width  # not inclusive
        
        # label window range
        y_start = x_end + (self.shift - self.input_width)
        y_end = y_start + self.label_width
        
        # slice input
        x = self.data[x_start:x_end]  # shape (input_width, num_features)
        
        # slice label
        if self.label_columns is not None:
            y = self.data[y_start:y_end, self.label_columns]
        else:
            y = self.data[y_start:y_end, :]
        
        # x shape => (input_width, num_features)
        # y shape => (label_width, num_label_features)
        return x, y
