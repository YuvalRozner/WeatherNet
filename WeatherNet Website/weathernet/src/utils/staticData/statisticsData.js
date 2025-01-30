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
