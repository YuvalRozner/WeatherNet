// TODO: update the content
export const getContent = (theme) => [
  {
    label: "About WeatherNet",
    description: `WeatherNet is an innovative weather forecasting platform designed to provide accurate temperature predictions for Israel, using advanced machine learning techniques.
    \nExplore how we revolutionize weather forecasting!\n`,
    image: "/figures/compressed_logo.png",
  },
  {
    label: "What is WeatherNet?",
    description: `The WeatherNet system consists of two primary components:
      \n\bThe backend:\b responsible for data processing, ML model training, and predictions.
      \bThe frontend:\b a user-friendly website for accessing forecasts.\n`,
    image: "system architecture macro",
  },
  {
    label: "Our Approach to Weather Forecasting",
    description: `Unlike traditional weather prediction models that rely on physics-based equations and statistical methods, WeatherNet harnesses the power of machine learning.
    \nWe collect real-time and historical weather data from the Israel Meteorological Service (IMS) across multiple stations, feeding it into our ML model to generate precise temperature forecasts for the coming days.\n`,
    image: "/figures/IMS_stations.png",
  },
  {
    label: "The ML Architecture",
    description: `We designed an architecture that first processes data through Convolutional Neural Networks (CNNs), where each feature is filtered independently.
    The output then transitions into a transformer phase, leveraging attention mechanisms to account for both temporal dependencies and geographical positioning.
    \nTo learn more about our architecture and how it enhances weather prediction, visit the 'Architecture' page.\n`,
    image: "The Architecture",
  },
  {
    label: "Accuracy & Performance",
    description: `WeatherNet has achieved high forecasting accuracy, with an error margin of less than 1.5°C for the majority of the data.
    \nWe trained four models with the same architecture structure, each specialized for a different forecast period: 0-12 hours, 12-24 hours, 24-36 hours, and 36-60 hours.
    \nLearn more about our performance metrics and comparisons with traditional forecasting models on our dedicated pages.\n`,
    image: `/figures/heatmap_forecast_error_${theme}.png`,
  },
  {
    label: "Technology Stack",
    description: `WeatherNet leverages cutting-edge technology, including Python, PyTorch, NumPy, Pandas, React, and Node.js for data processing, the ML model, and the web application UI.
    \nOur system is deployed using Firebase Hosting and Firebase Functions, ensuring seamless scalability and real-time updates for weather forecasting.\n`,
    image: "",
  },
  {
    label: "Project Workflow",
    description: `WeatherNet was originally developed as a final-year project by Yuval Rozner and Dor Shabat during Our Software Engineering degree. The project evolved through a structured workflow across two semesters.
    \nIn the first semester, we focused on learning the necessary subjects, conducting deep research on relevant topics, and establishing a connection with the Israel Meteorological Service (IMS). We developed a proof-of-concept mini-system to validate our approach and published our Phase A paper.
    \nDuring the second semester, we refined our approach by processing vast amounts of data, building our model, and fine-tuning hyperparameters for improved accuracy. Additionally, we developed the user interface and integrated the backend prediction module with the WeatherNet website. Finally, we successfully deployed a live version of WeatherNet, published our Phase B paper, and released both the User Manual and Developer Manual.\n`,
    image: `/figures/project_workflow_${theme}.png`,
  },
  {
    label: "Our Team & Partnerships",
    description: `WeatherNet was developed by Dor Shabat and Yuval Rozner, aiming to apply advanced machine learning techniques to improve weather forecasting accuracy.
    \nWe extend our gratitude to the Israel Meteorological Service (IMS) for providing invaluable weather data that powers our predictions, enabling us to refine and enhance our models.\n`,
    image: "/figures/weathernet_ims_together.png",
  },
];

// WeatherNet was originally developed as a final-year project by Yuval Rozner and Dor Shabat during their Software Engineering degree.
