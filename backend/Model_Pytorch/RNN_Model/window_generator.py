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

def main():
    import numpy as np

    # Generate sample data: 19 time steps with 3 features
    data = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1],
        [2.2, 2.3, 2.4],
        [2.5, 2.6, 2.7],
        [2.8, 2.9, 3.0],
        [3.1, 3.2, 3.3],
        [3.4, 3.5, 3.6],
        [3.7, 3.8, 3.9],
        [4.0, 4.1, 4.2],
        [4.3, 4.4, 4.5],
        [4.6, 4.7, 4.8],
        [4.9, 5.0, 5.1],
        [5.2, 5.3, 5.4],
        [5.5, 5.6, 5.7],
        [5.8, 5.9, 6.0]
    ])

    # Define WindowGenerator parameters
    input_width = 10
    label_width = 5
    shift = 1
    label_columns = [0, 2]  # Using first and third features as labels

    # Instantiate WindowGenerator
    window_gen = WindowGenerator(
        data=data,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        label_columns=label_columns
    )

    # Print the number of samples
    print(f"Number of samples: {len(window_gen)}")

    # Retrieve and print the first 2 samples
    for i in range(2):
        x, y = window_gen[i]
        print(f"\nSample {i+1}:")
        print("Input:", x)
        print("Label:", y)

if __name__ == "__main__":
    main()
