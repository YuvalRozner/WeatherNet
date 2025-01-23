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

        # Corrected number of samples
        self.num_samples = len(self.data) - self.total_window_size + 1

        if self.num_samples < 1:
            raise ValueError("Not enough data to create windows with these parameters.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # input window range
        x_start = idx
        x_end = x_start + self.input_width  # not inclusive

        # label window range
        y_start = x_start + self.total_window_size - self.label_width
        y_end = x_start + self.total_window_size

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

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input window size: {self.input_width}',
            f'Label window size: {self.label_width}',
            f'Shift: {self.shift}',
            f'Label columns indices: {self.label_columns}'
        ])
def test_window_generator():
    # Sample data: 100 time steps, 3 features
    data = np.arange(300).reshape(100, 3)
    input_width = 24
    label_width = 1
    shift = 1
    label_columns = [2]  # Assuming we're predicting the third feature

    dataset = WindowGenerator(
        data=data,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        label_columns=label_columns
    )

    print(dataset)

    # Fetch a sample
    x, y = dataset[0]
    print("Input shape:", x.shape)  # Expected: (24, 3)
    print("Label shape:", y.shape)  # Expected: (1, 1)
    print("Label value:", y.item())  # Expected: data[24, 2]

    # Verify alignment
    assert y.item() == data[24, 2], "Label alignment incorrect!"

    print("Test passed!")

test_window_generator()
