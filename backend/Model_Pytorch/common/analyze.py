import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


# https://chatgpt.com/share/6794d6f7-4e44-8010-91ed-30cf30a80b3a
def flatten_data(predictions, actuals):
    """
    Flattens nested lists of predictions and actuals into single-dimensional lists.

    Parameters:
    - predictions (list of lists): Nested list containing predicted temperatures.
    - actuals (list of lists): Nested list containing actual temperatures.

    Returns:
    - pd.DataFrame: DataFrame with flattened 'Predicted' and 'Actual' columns.
    """
    flat_predictions = [temp for window in predictions for temp in window]
    flat_actuals = [temp for window in actuals for temp in window]

    data = pd.DataFrame({
        'Predicted': flat_predictions,
        'Actual': flat_actuals
    })

    return data


def compute_error_metrics(actual, predicted):
    """
    Computes key error metrics between actual and predicted values.

    Parameters:
    - actual (array-like): Actual temperature values.
    - predicted (array-like): Predicted temperature values.

    Returns:
    - dict: Dictionary containing MAE, MSE, RMSE, MAPE, and R² Score.
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted) * 100  # Percentage
    r2 = r2_score(actual, predicted)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2_Score': r2
    }

    return metrics


def per_hour_analysis(predictions, actuals, num_hours=24):
    """
    Computes error metrics for each prediction hour.

    Parameters:
    - predictions (list of lists): Nested list of predicted temperatures.
    - actuals (list of lists): Nested list of actual temperatures.
    - num_hours (int): Number of hours predicted in each window.

    Returns:
    - pd.DataFrame: DataFrame containing MAE, RMSE, and MAPE for each hour.
    """
    num_windows = len(predictions)

    # Convert to NumPy arrays for reshaping
    predictions_array = np.array(predictions).reshape(-1, num_hours)
    actuals_array = np.array(actuals).reshape(-1, num_hours)

    # Create hour labels
    hour_columns = [f'Hour_{i + 1}' for i in range(num_hours)]

    # Flatten arrays and create DataFrame
    df_per_hour = pd.DataFrame({
        'Actual': actuals_array.flatten(),
        'Predicted': predictions_array.flatten(),
        'Hour': np.tile(hour_columns, num_windows)
    })

    # Compute metrics per hour
    per_hour_metrics = df_per_hour.groupby('Hour').apply(
        lambda x: pd.Series({
            'MAE': mean_absolute_error(x['Actual'], x['Predicted']),
            'RMSE': np.sqrt(mean_squared_error(x['Actual'], x['Predicted'])),
            'MAPE': mean_absolute_percentage_error(x['Actual'], x['Predicted']) * 100
        })
    ).reset_index()

    return per_hour_metrics


def plot_time_series(data, num_points=500):
    """
    Plots the actual and predicted temperatures over a specified number of points.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'Actual' and 'Predicted' columns.
    - num_points (int): Number of data points to plot.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(data['Actual'][:num_points], label='Actual', color='blue')
    plt.plot(data['Predicted'][:num_points], label='Predicted', color='orange', alpha=0.7)
    plt.title('Temperature Prediction vs Actuals')
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()


def plot_error_distribution(data):
    """
    Plots the distribution of prediction errors.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'Error' column.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Error'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (°C)')
    plt.ylabel('Frequency')
    plt.show()


def analyze(actuals, predictions, num_hours=1):
    data = flatten_data(predictions, actuals)

    # 2. Compute Error Metrics
    metrics = compute_error_metrics(data['Actual'], data['Predicted'])
    print("Error Metrics:")
    for key, value in metrics.items():
        if key == 'MAPE':
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.2f}")

    # 3. Per-Hour Analysis
    per_hour_metrics = per_hour_analysis(predictions, actuals, num_hours)
    print("\nPer-Hour Performance Metrics:")
    print(per_hour_metrics)

    # 4. Visualizations
    # a. Time Series Plot
    plot_time_series(data, num_points=500)

    # c. Error Distribution
    data['Error'] = data['Predicted'] - data['Actual']
    plot_error_distribution(data)


def plot_actual_vs_predicted_per_window(predictions, actuals, num_hours=24, max_windows=10):
    """
    Plots actual vs. predicted temperatures for each window with distinct colors.

    Parameters:
    - predictions (list of lists): Nested list containing predicted temperatures.
    - actuals (list of lists): Nested list containing actual temperatures.
    - num_hours (int): Number of hours predicted in each window.
    - max_windows (int): Maximum number of windows to plot for clarity.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from itertools import cycle

    # Limit the number of windows to plot for clarity
    if len(predictions) > max_windows:
        print(f"Plotting the first {max_windows} windows for clarity.")
        predictions = predictions[:max_windows]
        actuals = actuals[:max_windows]

    # Define a color palette
    color_cycle = iter(['red', 'green', 'blue', 'yellow'])  # Example color cycle

    plt.figure(figsize=(15, 8))

    for idx, (pred_window, actual_window) in enumerate(zip(predictions, actuals)):
        color = next(color_cycle)
        hours = range(idx, num_hours + idx)

        plt.plot(hours, actual_window, marker='o', linestyle='-', color=color, label=f'Actual Window {idx + 1}')
        plt.plot(hours, pred_window, marker='x', linestyle='--', color=color, label=f'Predicted Window {idx + 1}')

    plt.title('Actual vs Predicted Temperatures per Window')
    plt.xlabel('Hour')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Example Data (Replace these with your actual data)
    # For demonstration, let's create synthetic data
    np.random.seed(42)  # For reproducibility
    num_windows = 100  # Number of windows in validation data
    num_hours = 3  # Number of hours predicted per window
    actuals = [[20, 21, 22], [21, 22, 23], [22, 23, 24]]
    predictions = [[25, 26, 27], [26, 27, 28], [30, 30, 33]]

    # 1. Flatten Data
    data = flatten_data(predictions, actuals)

    # 2. Compute Error Metrics
    metrics = compute_error_metrics(data['Actual'], data['Predicted'])
    print("Error Metrics:")
    for key, value in metrics.items():
        if key == 'MAPE':
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.2f}")

    # 3. Per-Hour Analysis
    per_hour_metrics = per_hour_analysis(predictions, actuals, num_hours)

    print("\nPer-Hour Performance Metrics:")
    print(per_hour_metrics)

    # 4. Visualizations
    # a. Time Series Plot
    plot_time_series(data, num_points=500)
    plot_actual_vs_predicted_per_window(predictions, actuals, num_hours=num_hours)

    # c. Error Distribution
    data['Error'] = data['Predicted'] - data['Actual']
    plot_error_distribution(data)


if __name__ == "__main__":
    main()
