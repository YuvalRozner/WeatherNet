export const local_weathernetServer = "http://localhost:8080";
export const deployed_weathernetServer =
  "https://imsforecast-3nuc7rzvbq-uc.a.run.app";

export const request = (request, args) => {
  switch (request) {
    case "imsForecast":
      if (process.env.NODE_ENV === "development") {
        return fetch(`${local_weathernetServer}?cityId=${args.cityId}`);
      } else if (process.env.NODE_ENV === "production") {
        return fetchWeatherData(args.cityId);
      }
      break;
    default:
      return null;
  }
};

export const fetchWeatherData = async (cityId) => {
  try {
    const response = await fetch(
      `${local_weathernetServer}/imsForecast?cityId=${cityId}`
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
