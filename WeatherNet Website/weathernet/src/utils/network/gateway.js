export const local_weathernetServer = "http://localhost:8080";
export const deployed_weathernetServer_imsForecast =
  "https://getimsforecast-3nuc7rzvbq-ew.a.run.app";
export const deployed_weathernetServer_getImsTrueData =
  "https://getimstruedata-3nuc7rzvbq-ew.a.run.app";

export const getImsForecast = (cityId) => {
  if (process.env.NODE_ENV === "development") {
    return fetchWeatherData(`${local_weathernetServer}/imsForecast`, cityId);
  } else if (process.env.NODE_ENV === "production") {
    return fetchWeatherData(deployed_weathernetServer_imsForecast, cityId);
  }
};

export const getImsTrueData = (cityId) => {
  if (process.env.NODE_ENV === "development") {
    return fetchWeatherData(`${local_weathernetServer}/getImsTrueData`, cityId);
  } else if (process.env.NODE_ENV === "production") {
    return fetchWeatherData(deployed_weathernetServer_getImsTrueData, cityId);
  }
};

export const fetchWeatherData = async (requestUrl, cityId) => {
  try {
    const response = await fetch(`${requestUrl}?cityId=${cityId}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const jsonData = await response.json();
    return jsonData;
  } catch (error) {
    console.error("Error loading the JSON data:", error);
    throw error;
  }
};
