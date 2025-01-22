export const weathernetServer = "http://localhost:8080";

export const fetchWeatherData = async (cityId) => {
  try {
    const response = await fetch(`${weathernetServer}/imsForecast/${cityId}`);
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
