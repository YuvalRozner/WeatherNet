import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class WindowGeneratorMultipleStations(Dataset):
    """
    Creates sliding windows from a multi-station dataset.
    data: shape (T, num_stations, num_features)
    input_width (int): window size for input
    label_width (int): how many time steps to predict
    shift (int): how far ahead the prediction starts after the input
    label_columns (list[int] or None): indices of the columns used as labels
    target_station_idx (int): index of the target station
    """
    def __init__(
        self,
        data,
        input_width,
        label_width,
        shift,
        label_columns=None,
        target_station_idx=0,
    ):
        super().__init__()

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        self.data = data  # shape (T, num_stations, num_features)

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        # total window = input plus how far to shift plus label_width
        self.total_window_size = input_width + shift - 1 + label_width

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

        self.target_station_idx = target_station_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # input window range
        x_start = idx
        x_end = x_start + self.input_width  # not inclusive

        # label window range
        y_start = x_start + self.input_width + self.shift - 1
        y_end = y_start + self.label_width

        # slice input
        x = self.data[x_start:x_end]  # shape (input_width, num_stations, num_features)

        # slice label
        if self.label_columns is not None:
            y = self.data[y_start:y_end, self.target_station_idx, self.label_columns]  # [label_width, num_label_features]
        else:
            y = self.data[y_start:y_end, self.target_station_idx, :]  # [label_width, num_features]

        # Optionally, aggregate labels if predicting a single value per window
        y = y.mean(dim=0)  # [num_label_features]

        # Return x and y
        return x, y

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input window size: {self.input_width}',
            f'Label window size: {self.label_width}',
            f'Shift: {self.shift}',
            f'Label columns indices: {self.label_columns}',
            f'Target station index: {self.target_station_idx}'
        ])
    

if __name__ == "__main__":
    import numpy as np
    import torch

    # Number of time steps, stations, and features
    T = 1000
    num_stations = 5
    num_features = 10

    # Generate random data
    np.random.seed(42)
    data = np.random.randn(T, num_stations, num_features).astype(np.float32)

    # Define target station index
    target_station_idx = 0  # Change as needed

    # Create WindowGenerator instance
    window_size = 24
    label_size = 1
    shift = 1

    window_gen = WindowGeneratorMultipleStations(
        data=data,
        input_width=window_size,
        label_width=label_size,
        shift=shift,
        label_columns=[2],  # Set to specific feature indices if needed
        target_station_idx=target_station_idx
    )

    # Create DataLoader
    batch_size = 32
    train_loader = DataLoader(window_gen, batch_size=batch_size, shuffle=True)

    # Iterate through one batch
    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"x shape: {x.shape}")  # Expected: [batch_size, input_width, num_stations, num_features]
        print(f"y shape: {y.shape}")  # Expected: [batch_size, label_width]
        break  # Only show the first batch
