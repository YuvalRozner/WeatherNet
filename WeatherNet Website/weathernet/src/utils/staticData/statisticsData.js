// TODO: update the images.

export const getStatisticsData = (theme) => [
  {
    title: "Error per Percentage of Validation Data",
    description:
      "This figure illustrates the error (difference between predicted and actual values) across various percentages of validation data. \n\nThe x-axis represents the percentage of data, while the y-axis shows the error in degrees Celsius. \n\nThe graph indicates that the error remains low for the majority of the data, increasing significantly only for a small portion.",
    image: `/figures/combined_Error_per_Percentage_of_Data_${theme}.png`,
  },
  {
    title: "Distribution of Prediction Errors",
    description:
      "This graph illustrates the distribution of prediction errors across the entire validation set, comprising vast amount of samples. \n\nThe x-axis represents the error in degrees Celsius, while the y-axis shows the number of samples. \n\nThe distribution is centered around 0°C, indicating that most predictions are accurate, with errors symmetrically distributed around this point.",
    image: `/figures/all_models_error_distribution_${theme}.png`,
  },
  {
    title: "Forecast Error Heatmap by Prediction Interval",
    description:
      "Our forecast model predicts temperature at intervals from 1 to 60 hours. \n\nThis heatmap shows the forecast error across these intervals. \nRows represent prediction intervals, and columns indicate the percentage of validation data. \nColor intensity reflects error magnitude in degrees Celsius, with darker shades indicating higher errors. \n\nThis helps understand how forecast accuracy varies with different intervals.",
    image: `/figures/heatmap_forecast_error_${theme}.png`,
  },
];

// TODO: update the data.
export const getmodelMetricsData = () => [
  {
    type: "MAE",
    models: {
      "1-12": 1.04967,
      "13-24": 1.432691,
      "25-36": 1.04967,
      "37-60": 1.432691,
    },
    overall: 0.5,
    tooltip: "Mean Absolute Error",
    explanation:
      "Mean Absolute Error (MAE) is a measure of the average magnitude of the errors in a set of predictions, without considering their direction. It is the average of the absolute differences between prediction and actual observation values.",
  },
  {
    type: "MSE",
    models: {
      "1-12": 2.138306,
      "13-24": 3.931717,
      "25-36": 2.138306,
      "37-60": 3.931717,
    },
    overall: 0.5,
    tooltip: "Mean Squared Error",
    explanation:
      "Mean Squared Error (MSE) is a measure of the average of the squares of the errors, or deviations, between predicted and actual observation values. It is the average of the squared differences between prediction and actual observation values.",
  },
  {
    type: "RMSE",
    models: {
      "1-12": 1.462295,
      "13-24": 1.982856,
      "25-36": 1.462295,
      "37-60": 1.982856,
    },
    overall: 0.5,
    tooltip: "Root Mean Squared Error",
    explanation:
      "Root Mean Squared Error (RMSE) is the square root of the mean of the squares of the errors, or deviations, between predicted and actual observation values. It is the square root of the average of the squared differences between prediction and actual observation values.",
  },
  {
    type: "MAPE",
    models: {
      "1-12": 6.038863,
      "13-24": 8.310858,
      "25-36": 6.038863,
      "37-60": 8.310858,
    },
    overall: 0.5,
    tooltip: "Mean Absolute Percentage Error",
    explanation:
      "Mean Absolute Percentage Error (MAPE) is a measure of the average of the absolute percentage differences between predicted and actual observation values. It is the average of the absolute percentage differences between prediction and actual observation values.",
  },
  {
    type: "R2_Score",
    models: {
      "1-12": 0.963163,
      "13-24": 0.932288,
      "25-36": 0.963163,
      "37-60": 0.932288,
    },
    overall: 0.5,
    tooltip: "R2 Score",
    explanation:
      "R2 Score (R2) is a measure of the proportion of the variance in the dependent variable that is predictable from the independent variable. It is the square of the correlation coefficient between the predicted and actual observation values.",
  },
];
