import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Load Data
data = pd.read_csv('weather_data.csv')  # Replace with your data source

# Handle missing values (e.g., fill with mean)
data.fillna(data.mean(), inplace=True)

# Remove duplicate records
data.drop_duplicates(inplace=True)

# Extract time-based features
if 'DateTime' in data.columns:
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['Hour'] = data['DateTime'].dt.hour
    data['Day'] = data['DateTime'].dt.day
    data['Month'] = data['DateTime'].dt.month
    data['Year'] = data['DateTime'].dt.year
    data.drop('DateTime', axis=1, inplace=True)

# Create lag features (e.g., previous hour's temperature)
data['Temp_Lag1'] = data['Temperature'].shift(1)
data['Temp_Lag2'] = data['Temperature'].shift(2)

# Drop rows with NaN values resulting from lagging
data.dropna(inplace=True)

# One-Hot Encode categorical variables (e.g., 'Weather')
if 'Weather' in data.columns:
    data = pd.get_dummies(data, columns=['Weather'], drop_first=True)

# Define features and target
X = data.drop('Temperature', axis=1)
y = data['Temperature']

# Split data into training and testing sets without shuffling (to maintain temporal order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Initialize scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit scalers on training data and transform
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Reshape target variables for scaling
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Scale targets
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Save scalers for future use (inference)
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

class WeatherDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets
train_dataset = WeatherDataset(X_train_scaled, y_train_scaled)
test_dataset = WeatherDataset(X_test_scaled, y_test_scaled)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TemperaturePredictor(nn.Module):
    def __init__(self, input_dim):
        super(TemperaturePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output layer for regression
        )
    
    def forward(self, x):
        return self.model(x)

# Determine input dimension
input_dim = X_train_scaled.shape[1]

# Initialize model
model = TemperaturePredictor(input_dim)

# Define loss function (Mean Squared Error for regression)
criterion = nn.MSELoss()

# Define optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training parameters
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(test_loader.dataset)
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

model.eval()
test_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item() * batch_X.size(0)
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}')

torch.save(model.state_dict(), 'temperature_predictor.pth')

def predict_temperature(new_data):
    """
    Predict temperature for new data.

    Parameters:
    - new_data: pandas DataFrame with the same features as the training data.

    Returns:
    - Predicted temperature in the original scale.
    """
    # Load scalers
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    
    # Handle missing values
    new_data.fillna(data.mean(), inplace=True)
    
    # Feature Engineering
    if 'DateTime' in new_data.columns:
        new_data['DateTime'] = pd.to_datetime(new_data['DateTime'])
        new_data['Hour'] = new_data['DateTime'].dt.hour
        new_data['Day'] = new_data['DateTime'].dt.day
        new_data['Month'] = new_data['DateTime'].dt.month
        new_data['Year'] = new_data['DateTime'].dt.year
        new_data.drop('DateTime', axis=1, inplace=True)
    
    # Create lag features
    # Note: Creating lag features requires historical data. Ensure 'Temp_Lag1' and 'Temp_Lag2' are provided.
    # If not, you need to fetch the latest temperatures from your data source.
    if 'Temp_Lag1' not in new_data.columns or 'Temp_Lag2' not in new_data.columns:
        raise ValueError("Lag features 'Temp_Lag1' and 'Temp_Lag2' must be provided in new_data.")
    
    # Encode categorical variables
    if 'Weather' in new_data.columns:
        new_data = pd.get_dummies(new_data, columns=['Weather'], drop_first=True)
    
    # Align new_data with training features
    # Any missing columns are filled with 0
    X_train_columns = X.columns
    new_data = new_data.reindex(columns=X_train_columns, fill_value=0)
    
    # Scale features
    new_data_scaled = scaler_X.transform(new_data)
    new_data_scaled = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)
    
    # Load model
    model = TemperaturePredictor(input_dim)
    model.load_state_dict(torch.load('temperature_predictor.pth'))
    model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        scaled_pred = model(new_data_scaled)
    
    # Convert prediction to CPU and NumPy
    scaled_pred = scaled_pred.cpu().numpy()
    
    # Inverse scale the prediction
    temp_pred = scaler_y.inverse_transform(scaled_pred)
    
    return temp_pred.flatten()

# Example new data
new_weather_data = pd.DataFrame({
    'Feature1': [value1],  # Replace with actual feature names and values
    'Feature2': [value2],
    # ... other features ...
    'Temp_Lag1': [previous_temp1],
    'Temp_Lag2': [previous_temp2],
    # 'Weather': ['Sunny'],  # Include if applicable
})

# Predict temperature
predicted_temperature = predict_temperature(new_weather_data)
print(f'Predicted Temperature: {predicted_temperature[0]:.2f}')
