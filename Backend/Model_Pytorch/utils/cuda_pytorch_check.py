import torch
import torch.nn as nn
import torch.optim as optim
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU for PyTorch.")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. You can only use CPU for PyTorch.")

# Simple model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out



if __name__ == "__main__":
    #check_cuda()
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model
    model = SimpleLSTM(input_size=10, hidden_size=20, num_layers=2, output_size=1).to(device)
    print(next(model.parameters()).device)

    # Dummy data
    x = torch.randn(64, 24, 10).to(device)  # (batch_size, sequence_length, input_size)
    y = torch.randn(64, 1).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print("Dummy training step completed.")
