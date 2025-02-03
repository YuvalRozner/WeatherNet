export const getStatisticsData = (theme) => [
  {
    title: "Forecast Error Heatmap by Prediction Interval",
    description:
      "Our forecast model predicts temperature at intervals from 1 to 60 hours. \n\nThis heatmap shows the forecast error across these intervals. \nRows represent prediction intervals, and columns indicate the percentage of validation data. \nColor intensity reflects error magnitude in degrees Celsius, with darker shades indicating higher errors. \n\nThis helps understand how forecast accuracy varies with different intervals.",
    image: `/figures/heatmap_forecast_error_${theme}.png`,
  },
  {
    title: "Distribution of Prediction Errors",
    description:
      "This graph illustrates the distribution of prediction errors across the entire validation set, comprising vast amount of samples. \n\nThe x-axis represents the error in degrees Celsius, while the y-axis shows the number of samples. \n\nThe distribution is centered around 0°C, indicating that most predictions are accurate, with errors symmetrically distributed around this point.",
    image: `/figures/all_models_error_distribution_${theme}.png`,
  },
  {
    title: "Error per Percentage of Validation Data",
    description:
      "This figure illustrates the error (difference between predicted and actual values) across various percentages of validation data. \n\nThe x-axis represents the percentage of data, while the y-axis shows the error in degrees Celsius. \n\nThe graph indicates that the error remains low for the majority of the data, increasing significantly only for a small portion.",
    image: `/figures/combined_Error_per_Percentage_of_Data_${theme}.png`,
  },
];

export const getStatisticsModelsGraphsData = (theme) => [
  {
    title: "model 1-12 hours",
    description: "model 1-12 hours",
    image: `/figures/time_series_model0_${theme}.png`,
  },
  {
    title: "model 12-24 hours",
    description: "model 12-24 hours",
    image: `/figures/time_series_model1_${theme}.png`,
  },
  {
    title: "model 22-36 hours",
    description: "model 24-36 hours",
    image: `/figures/time_series_model2_${theme}.png`,
  },
  {
    title: "model 36-60 hours",
    description: "model 36-60 hours",
    image: `/figures/time_series_model3_${theme}.png`,
  },
];

export const getmodelMetricsData = () => [
  {
    type: "MAE",
    models: {
      "1-12": 0.958081564,
      "13-24": 1.40994894893669,
      "25-36": 1.74030704221972,
      "37-60": 1.94939084908824,
    },
    overall: 1.51443210111225,
    tooltip: "Mean Absolute Error",
    explanation:
      "Mean Absolute Error (MAE) is a measure of the average magnitude of the errors in a set of predictions, without considering their direction. It is the average of the absolute differences between prediction and actual observation values.",
  },
  {
    type: "MSE",
    models: {
      "1-12": 1.87332171733182,
      "13-24": 3.88104868845325,
      "25-36": 5.78592471411383,
      "37-60": 7.31854670829491,
    },
    overall: 4.71471045704846,
    tooltip: "Mean Squared Error",
    explanation:
      "Mean Squared Error (MSE) is a measure of the average of the squares of the errors, or deviations, between predicted and actual observation values. It is the average of the squared differences between prediction and actual observation values.",
  },
  {
    type: "RMSE",
    models: {
      "1-12": 1.36869343438617,
      "13-24": 1.97003773782464,
      "25-36": 2.40539491853496,
      "37-60": 2.7052812623265,
    },
    overall: 2.17133840224145,
    tooltip: "Root Mean Squared Error",
    explanation:
      "Root Mean Squared Error (RMSE) is the square root of the mean of the squares of the errors, or deviations, between predicted and actual observation values. It is the square root of the average of the squared differences between prediction and actual observation values.",
  },
  {
    type: "MAPE",
    models: {
      "1-12": 5.45281695121325,
      "13-24": 8.08944810109361,
      "25-36": 10.0409439430557,
      "37-60": 12.0256661452063,
    },
    overall: 8.90221878514224,
    tooltip: "Mean Absolute Percentage Error",
    explanation:
      "Mean Absolute Percentage Error (MAPE) is a measure of the average of the absolute percentage differences between predicted and actual observation values. It is the average of the absolute percentage differences between prediction and actual observation values.",
  },
  {
    type: "R2_Score",
    models: {
      "1-12": 0.967697989873033,
      "13-24": 0.93313506201016,
      "25-36": 0.90039383864744,
      "37-60": 0.878830696879185,
    },
    overall: 0.919609250579684,
    tooltip: "R2 Score",
    explanation:
      "R2 Score (R2) is a measure of the proportion of the variance in the dependent variable that is predictable from the independent variable. It is the square of the correlation coefficient between the predicted and actual observation values.",
  },
];
