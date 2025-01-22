export const local_weathernetServer = "http://localhost:8080";
export const deployed_weathernetServer =
  "https://imsforecast-3nuc7rzvbq-uc.a.run.app";

export const fetchWeatherData = async (cityId) => {
  try {
    const response = await fetch(
      `${deployed_weathernetServer}?cityId=${cityId}`
    );
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
