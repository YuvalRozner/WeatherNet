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
import os

def compute_error_metrics(actual, predicted):
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
    print("compute_error_metrics completed.")
    return metrics

def per_hour_analysis(predictions_series, actuals_series, label_width=24, start_hour=0):
    # Ensure the length is divisible by label_width
    num_windows = len(predictions_series) // label_width

    # Reshape into 2D NumPy arrays
    predictions_array = predictions_series.values.reshape(num_windows, label_width)
    actuals_array = actuals_series.values.reshape(num_windows, label_width)

    # Create hour labels
    hour_columns = [f'Hour_{start_hour + i + 1}' for i in range(label_width)]

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

    per_hour_metrics['Hour_Index'] = per_hour_metrics['Hour'].str.extract(r'(\d+)').astype(int)
    per_hour_metrics = per_hour_metrics.sort_values(by='Hour_Index').drop(columns=['Hour_Index'])

    print("per_hour_analysis completed.")
    return per_hour_metrics

#################################################### plots ####################################################

def plot_error_distribution(data, model_txt, path_to_save, percentages=False, theme='both'):
    sns.reset_orig()

    # Define themes to process
    themes = ['light', 'dark'] if theme == 'both' else [theme]

    for current_theme in themes:
        # Define colors based on current_theme
        if (current_theme == 'dark'):
            text_color = 'white'
            face_color = '#282c34'
            grid = False
            tick_color = 'white'
            edge_color = 'white'
            palette = 'viridis'  # Suitable for dark backgrounds
            title_suffix = 'dark'
        else:
            text_color = 'black'
            face_color = 'white'
            grid = True
            tick_color = 'black'
            edge_color = 'black'
            palette = 'coolwarm'  # Suitable for light backgrounds
            title_suffix = 'light'

        # Determine the save path
        if theme == 'both':
            # Insert the theme suffix before the file extension
            base, ext = os.path.splitext(path_to_save)
            save_path = f"{base}_{title_suffix}{ext}"
        else:
            save_path = path_to_save

        # Create figure with specified facecolor
        plt.figure(figsize=(10, 5), facecolor=face_color)

        # Choose the statistic for the histogram
        stat = "percent" if percentages else "count"

        # Create histogram
        sns.histplot(
            data['Error'],
            bins=50,
            kde=True,
            color='purple',
            stat=stat,
            edgecolor=edge_color
        )

        # Set labels based on the type of plot
        ylabel = 'Percentage (%)' if percentages else 'Frequency'

        # Customize titles and labels
        plt.title(
            f'Distribution of Prediction Errors in {model_txt}, from {len(data["Error"])} samples',
            color=text_color
        )
        plt.xlabel('Error (°C)', color=text_color)
        plt.ylabel(ylabel, color=text_color)

        # Customize tick labels
        ax = plt.gca()
        ax.tick_params(colors=tick_color)

        # Change the background color of axes
        ax.set_facecolor(face_color)

        # Remove or show grid lines based on theme
        if grid:
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
        else:
            ax.grid(False)

        # Change spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)

        # Change tick label colors explicitly
        plt.setp(ax.get_xticklabels(), color=tick_color)
        plt.setp(ax.get_yticklabels(), color=tick_color)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(save_path, facecolor=plt.gcf().get_facecolor())
        plt.close()
        print(f"Saved error distribution plot to {save_path}")

def plot_time_series(data, graph_title, path_to_save, theme='both'):
    sns.reset_orig()

    # Define themes to process
    themes = ['light', 'dark'] if theme == 'both' else [theme]

    for current_theme in themes:
        # Define theme-specific settings
        if current_theme == 'dark':
            text_color = 'white'
            face_color = '#282c34'
            grid = False
            tick_color = 'white'
            spine_color = 'white'
            legend_color = 'white'
            theme_suffix = 'dark'
        else:
            text_color = 'black'
            face_color = 'white'
            grid = True
            tick_color = 'black'
            spine_color = 'black'
            legend_color = 'black'
            theme_suffix = 'light'

        # Determine the save path
        if theme == 'both':
            # Insert the theme suffix before the file extension
            base, ext = os.path.splitext(path_to_save)
            save_path = f"{base}_{theme_suffix}{ext}"
        else:
            save_path = path_to_save

        # Create figure with specified facecolor
        plt.figure(figsize=(15, 5), facecolor=face_color)

        # Plot Actual and Predicted
        plt.plot(data['Actual'], label='Actual', color='blue')
        plt.plot(data['Predicted'], label='Predicted', color='orange', alpha=0.7)

        # Customize titles and labels
        plt.title(graph_title, color=text_color)
        plt.xlabel('Forecasted Time (hours) (concat)', color=text_color)
        plt.ylabel('Temperature (°C)', color=text_color)

        # Customize tick labels
        ax = plt.gca()
        ax.tick_params(colors=tick_color)

        # Change the background color of axes
        ax.set_facecolor(face_color)

        # Remove or show grid lines based on theme
        if grid:
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
        else:
            ax.grid(False)

        # Change spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)

        # Change tick label colors explicitly
        plt.setp(ax.get_xticklabels(), color=tick_color)
        plt.setp(ax.get_yticklabels(), color=tick_color)

        # Customize legend
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color(legend_color)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(save_path, facecolor=plt.gcf().get_facecolor())
        plt.close()
        print(f"Saved time series plot to {save_path}")


def create_heatmap(dfs, graph_title, path_to_save, theme='both'):
    sns.reset_orig()

    # Validate the 'theme' parameter
    valid_themes = ['light', 'dark', 'both']
    if theme not in valid_themes:
        raise ValueError(f"Invalid theme '{theme}'. Supported themes are: {valid_themes}")

    # Define themes to process
    themes_to_plot = ['light', 'dark'] if theme == 'both' else [theme]

    for current_theme in themes_to_plot:
        # Define theme-specific settings
        if current_theme == 'dark':
            text_color = 'white'
            face_color = '#282c34'
            grid = False
            cmap = 'coolwarm'  # Suitable palette for dark backgrounds
            line_color = 'white'
            title_suffix = 'dark'
            tick_color = 'white'
            spine_color = 'white'
            cbar_tick_color = 'white'
            cbar_label_color = 'white'
        else:
            text_color = 'black'
            face_color = 'white'
            grid = True
            cmap = 'coolwarm'  # Suitable palette for light backgrounds
            line_color = 'black'
            title_suffix = 'light'
            tick_color = 'black'
            spine_color = 'black'
            cbar_tick_color = 'black'
            cbar_label_color = 'black'

        # Determine the save path
        if theme == 'both':
            base, ext = os.path.splitext(path_to_save)
            save_path = f"{base}_{title_suffix}{ext}"
        else:
            save_path = path_to_save

        # Stack all 'Error' values into a 2D array (each row represents a df's errors)
        error_values = np.array([df['Error'].values for df in dfs])

        # Extract percentiles from the first DataFrame (assuming they are consistent across all)
        percentiles = dfs[0]['Percentile'].tolist()

        # Create figure with specified facecolor
        plt.figure(figsize=(12, 1.5 * len(dfs)), facecolor=face_color)

        # Create heatmap
        ax = sns.heatmap(
            error_values,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            xticklabels=percentiles,
            yticklabels=[f"Model {i}" for i in range(len(dfs))],
            cbar=True,
            linewidths=0.5,
            linecolor='gray' if current_theme == 'light' else 'white'
        )

        # Customize titles and labels based on theme
        plt.title(graph_title, color=text_color, fontsize=16)
        plt.xlabel('Percentile', color=text_color)
        plt.ylabel('Model Index', color=text_color)

        # Customize tick labels
        ax.tick_params(colors=tick_color)

        # Change the background color of axes
        ax.set_facecolor(face_color)

        # Change spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)

        # Change tick label colors explicitly
        plt.setp(ax.get_xticklabels(), color=tick_color)
        plt.setp(ax.get_yticklabels(), color=tick_color)

        # Adjust color bar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(color=cbar_tick_color)
        cbar.ax.yaxis.label.set_color(cbar_label_color)
        cbar.ax.tick_params(labelsize=10, labelcolor=cbar_tick_color)

        # Remove or show grid lines based on theme
        if grid:
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
        else:
            ax.grid(False)

        # Change spine colors again if needed (optional)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(save_path, facecolor=plt.gcf().get_facecolor())
        plt.close()
        print(f"Saved heatmap to {save_path}")


def plot_error_bar(df, graph_title, path_to_save, theme='both'):
    sns.reset_orig()

    # Validate the 'theme' parameter
    valid_themes = ['light', 'dark', 'both']
    if theme not in valid_themes:
        raise ValueError(f"Invalid theme '{theme}'. Supported themes are: {valid_themes}")

    # Define themes to process
    themes_to_plot = ['light', 'dark'] if theme == 'both' else [theme]

    for current_theme in themes_to_plot:
        # Define theme-specific settings
        if current_theme == 'dark':
            text_color = 'white'
            face_color = '#282c34'
            grid = False
            tick_color = 'white'
            spine_color = 'white'
            palette = 'viridis'  # Suitable palette for dark backgrounds
            title_suffix = 'dark'
            label_color = 'white'
            annotate_color = 'white'
        else:
            text_color = 'black'
            face_color = 'white'
            grid = True
            tick_color = 'black'
            spine_color = 'black'
            palette = 'viridis'  # Can choose a different palette if desired
            title_suffix = 'light'
            label_color = 'black'
            annotate_color = 'black'

        # Determine the save path
        if theme == 'both':
            # Insert the theme suffix before the file extension
            base, ext = os.path.splitext(path_to_save)
            save_path = f"{base}_{title_suffix}{ext}"
        else:
            save_path = path_to_save

        # Create figure with specified facecolor
        plt.figure(figsize=(12, 6), facecolor=face_color)

        # Create bar plot
        ax = sns.barplot(x='Percentile', y='Error', data=df, palette=palette,
                         edgecolor='black' if current_theme == 'light' else 'white')

        # Customize titles and labels
        plt.title(graph_title, color=text_color, fontsize=16)
        plt.xlabel('Percentile', color=label_color, fontsize=14)
        plt.ylabel('Error', color=label_color, fontsize=14)

        # Customize tick labels
        ax.tick_params(colors=tick_color)
        plt.xticks(rotation=45, color=tick_color)
        plt.yticks(color=tick_color)

        # Change the background color of axes
        ax.set_facecolor(face_color)

        # Remove or show grid lines based on theme
        if grid:
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
        else:
            ax.grid(False)

        # Change spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)

        # Remove top and right spines for a cleaner look
        sns.despine()

        # Add error labels on top of each bar with contrasting color
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, color=annotate_color,
                        xytext=(0, 5),  # Offset text by 5 points above the bar
                        textcoords='offset points')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(save_path, facecolor=plt.gcf().get_facecolor())
        plt.close()
        print(f"Saved error bar plot to {save_path}")

#################################################### main ####################################################

def prepare_data(dfs):
    additional_percentiles = [
        0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95
    ]
    percentile_dfs = []
    for df in dfs:
        # Process the 'Error' column
        df['Error_abs_round_2'] = df['Error'].abs().round(2)

        # Calculate the percentile values
        percentile_values = df['Error_abs_round_2'].quantile(additional_percentiles).sort_index()

        # Create a new DataFrame with Percentile and Error columns
        percentile_df = pd.DataFrame({
            'Percentile': [f"{int(p*100)}%" for p in percentile_values.index],
            'Error': percentile_values.values
        })

        # Optionally, reset the index if you prefer default integer indexing
        percentile_df.reset_index(drop=True, inplace=True)
        percentile_dfs.append(percentile_df)
    print("prepare_data completed.")
    return percentile_dfs


if __name__ == "__main__":
    # Define paths
    path_to_file = os.path.abspath(os.path.dirname(__file__))  
    folder_path = os.path.join(path_to_file, '..', 'AdvancedModel', 'output', 'analyze_output')
    folder_path_to_save = os.path.join(folder_path, 'analyze_the_hell_out_of_it')  # Corrected folder name

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path_to_save):
        os.makedirs(folder_path_to_save)
    else:
        print(f"Folder {folder_path_to_save} already exists")
        exit(1)

    # Read all CSV files in the folder
    dfs = []
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            dfs.append(df)
            os.makedirs(os.path.join(folder_path_to_save, f"model_{i}"), exist_ok=True)
    if not dfs:
        print("No CSV files found in the specified folder.")
        exit(1)

    # Combine all DataFrames
    dfs_combined = pd.concat(dfs)
    dfs_combined.reset_index(drop=True, inplace=True)

    # Prepare percentile data
    percentile_dfs = prepare_data(dfs)
    percentile_dfs_combined = prepare_data([dfs_combined])[0]

    # Save percentile metrics to CSV
    for i, percentile_df in enumerate(percentile_dfs):
        plot_error_bar(
            df=percentile_df,
            graph_title=f"Error per Percentage of Data. Model: {i}",
            path_to_save=os.path.join(folder_path_to_save, f"model_{i}","Error_per_Percentage_of_Data.png"),
            theme='both')
        print(f"Plotted error bar for model {i}")

    # Plot and save combined graphs
    plot_error_bar(
        df=percentile_dfs_combined,
        graph_title="Error per Percentage of Data. Combined Models",
        path_to_save=os.path.join(folder_path_to_save, "combined_Error_per_Percentage_of_Data.png"),
        theme='both'  # Generates both light and dark themed plots
    )

    # Create and save heatmaps
    create_heatmap(
        dfs=percentile_dfs,
        graph_title='Forecast Error per Percentage of Data',
        path_to_save=os.path.join(folder_path_to_save, "heatmap_forecast_error.png"),
        theme='both'  # Generates both light and dark themed heatmaps
    )

    start_hour = 0
    metrics_list = []
    per_hour_metrics_list = []

    for i, df in enumerate(dfs):
        # Perform per-hour analysis
        label_width = int(df['label_width'].iloc[0]) if 'label_width' in df.columns else 24  # Default to 24 if not present
        per_hour_metrics = per_hour_analysis(df['Predicted'], df['Actual'], label_width=label_width, start_hour=start_hour)
        start_hour += label_width

        per_hour_metrics_list.append(per_hour_metrics)

        # Compute overall error metrics
        metrics = compute_error_metrics(df['Actual'], df['Predicted'])
        metrics_df = pd.DataFrame([metrics])
        metrics_list.append(metrics_df)

        # Plot and save time series
        plot_time_series(
            data=df,
            graph_title=f"Predicted vs Actuals in Model {i}",
            path_to_save=os.path.join(folder_path_to_save, f"model_{i}","time_series.png"),
            theme='both')

        # Plot and save error distributions
        plot_error_distribution(
            data=df,
            model_txt=f"Model {i}",
            path_to_save=os.path.join(folder_path_to_save, f"model_{i}","error_distribution_percentages.png"),
            percentages=True,
            theme='both')
        plot_error_distribution(
            data=df,
            model_txt=f"Model {i}",
            path_to_save=os.path.join(folder_path_to_save, f"model_{i}","error_distribution.png"),
            percentages=False,
            theme='both')

    # Save combined error metrics
    combined_metrics_df = pd.concat(metrics_list, ignore_index=True)
    combined_metrics_df.to_csv(os.path.join(folder_path_to_save, "combined_error_metrics.csv"), index=False)

    if per_hour_metrics_list:
        combined_per_hour_metrics = pd.concat(per_hour_metrics_list, ignore_index=True)
        combined_per_hour_metrics.to_csv(os.path.join(folder_path_to_save, "combined_per_hour_metrics.csv"), index=False)

    # Compute and save combined error distributions
    plot_error_distribution(
        data=dfs_combined,
        model_txt="All Models",
        path_to_save=os.path.join(folder_path_to_save, "all_models_error_distribution_percentages.png"),
        percentages=True,
        theme = 'both')
    plot_error_distribution(
        data=dfs_combined,
        model_txt="All Models",
        path_to_save=os.path.join(folder_path_to_save, "all_models_error_distribution.png"),
        percentages=False,
        theme = 'both')

    # Compute and save combined error metrics
    combined_overall_metrics = compute_error_metrics(dfs_combined['Actual'], dfs_combined['Predicted'])
    combined_overall_metrics_df = pd.DataFrame([combined_overall_metrics])
    combined_overall_metrics_df.to_csv(os.path.join(folder_path_to_save, "all_models_error_metrics.csv"), index=False)
    print("All plots have been saved successfully.")
