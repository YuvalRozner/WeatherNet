# main.py

from Backend.Model_Pytorch.common.window_generator import WindowGenerator
from Backend.Model_Pytorch.common.data import normalize_data, load_pkl_file
from Backend.Model_Pytorch.RNN_Model_our_data.model import LSTMModel
from Backend.Model_Pytorch.RNN_Model_our_data.train import train_model
from Backend.Model_Pytorch.RNN_Model_our_data.parameters import PARAMS, WINDOW_PARAMS, LSTM_MODEL_PARAMS
import os


if __name__ == "__main__":

    # 1) Load data
    df = load_pkl_file(PARAMS['fileName'])
    # 2) Convert to numpy array
    data_np = df.values  # shape (T, in_channels)
    
    # 3) Create train/val split
    train_size = int(0.8 * len(data_np))
    train_data = data_np[:train_size]
    val_data = data_np[train_size:]
    
    # 4) Normalize the data
    train_data_scaled, val_data_scaled, scaler = normalize_data(train_data, val_data, scaler_path=os.path.join(os.path.dirname(__file__),
                                                                                                               'output', 'scaler.pkl'))

    print("split data and normilized")

    # 5) Create Datasets
    input_width = WINDOW_PARAMS['input_width']
    label_width = WINDOW_PARAMS['label_width']
    shift = WINDOW_PARAMS['shift']

    column_indices = {name: i for i, name in enumerate(df.columns)}
    label_columns = [column_indices[WINDOW_PARAMS['label_columns'][0]]]
    
    train_dataset = WindowGenerator(train_data_scaled, input_width, label_width, shift, label_columns)
    val_dataset   = WindowGenerator(val_data_scaled,   input_width, label_width, shift, label_columns)
    
    # 6) Instantiate model (LSTM)
    in_channels = df.shape[1]  # number of features

    device = PARAMS['device']
    print(f"Using device: {device}")

    model = LSTMModel(
        in_channels=in_channels,
        hidden_dim=LSTM_MODEL_PARAMS['hidden_dim'],
        num_layers=LSTM_MODEL_PARAMS['num_layers'],
        label_width=label_width
    )

    # 7) Train
    train_model(
        train_dataset,
        val_dataset,
        model,
        epochs=PARAMS['epochs'],
        batch_size=32,
        lr=1e-5,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), 'output', 'checkpoints'),
        resume=PARAMS['resume'],
        device=device
    )
