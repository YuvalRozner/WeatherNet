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
      "1-12": 0.957917256613646,
      "13-24": 1.40984387475643,
      "25-36": 1.41232184725876,
      "37-60": 1.94909789927995,
    },
    overall: 1.4322952194772,
    tooltip: "Mean Absolute Error",
    explanation:
      "Mean Absolute Error (MAE) is a measure of the average magnitude of the errors in a set of predictions, without considering their direction. It is the average of the absolute differences between prediction and actual observation values.",
  },
  {
    type: "MSE",
    models: {
      "1-12": 1.87285153437129,
      "13-24": 3.88021908539729,
      "25-36": 3.89498489236451,
      "37-60": 7.3166124521578,
    },
    overall: 4.24116699107272,
    tooltip: "Mean Squared Error",
    explanation:
      "Mean Squared Error (MSE) is a measure of the average of the squares of the errors, or deviations, between predicted and actual observation values. It is the average of the squared differences between prediction and actual observation values.",
  },
  {
    type: "RMSE",
    models: {
      "1-12": 1.36852166017615,
      "13-24": 1.9698271714537,
      "25-36": 1.97357160811674,
      "37-60": 2.70492374239234,
    },
    overall: 2.05940937918441,
    tooltip: "Root Mean Squared Error",
    explanation:
      "Root Mean Squared Error (RMSE) is the square root of the mean of the squares of the errors, or deviations, between predicted and actual observation values. It is the square root of the average of the squared differences between prediction and actual observation values.",
  },
  {
    type: "MAPE",
    models: {
      "1-12": 5.45277105887534,
      "13-24": 8.09048062749564,
      "25-36": 8.11832601892737,
      "37-60": 12.0266230064599,
    },
    overall: 8.42205017793956,
    tooltip: "Mean Absolute Percentage Error",
    explanation:
      "Mean Absolute Percentage Error (MAPE) is a measure of the average of the absolute percentage differences between predicted and actual observation values. It is the average of the absolute percentage differences between prediction and actual observation values.",
  },
  {
    type: "R2_Score",
    models: {
      "1-12": 0.967722942678878,
      "13-24": 0.933177153642848,
      "25-36": 0.9329228656172,
      "37-60": 0.878923271281989,
    },
    overall: 0.927703529814378,
    tooltip: "R2 Score",
    explanation:
      "R2 Score (R2) is a measure of the proportion of the variance in the dependent variable that is predictable from the independent variable. It is the square of the correlation coefficient between the predicted and actual observation values.",
  },
];
