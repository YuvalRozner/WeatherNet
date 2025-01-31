export const local_weathernetServer = "http://localhost:8080";
export const deployed_weathernetServer_imsForecast =
  "https://getimsforecast-3nuc7rzvbqY-ew.a.run.app";
export const deployed_weathernetServer_getImsTrueData =
  "https://getimstruedata-3nuc7rzvbqY-ew.a.run.app";

export const getImsForecast = async (cityId) => {
  try {
    if (process.env.NODE_ENV === "development") {
      return await fetchWeatherData(
        `${local_weathernetServer}/imsForecast`,
        cityId
      );
    } else if (process.env.NODE_ENV === "production") {
      return await fetchWeatherData(
        deployed_weathernetServer_imsForecast,
        cityId
      );
    }
  } catch (error) {
    console.error(
      "Primary server fetch failed, trying the alternative server:",
      error
    );
    try {
      if (process.env.NODE_ENV === "development") {
        return await fetchWeatherData(
          deployed_weathernetServer_imsForecast,
          cityId
        );
      } else if (process.env.NODE_ENV === "production") {
        return await fetchWeatherData(
          `${local_weathernetServer}/imsForecast`,
          cityId
        );
      }
    } catch (secondaryError) {
      console.error("Both server fetch attempts failed:", secondaryError);
      throw secondaryError;
    }
  }
};

export const getImsTrueData = async (cityId) => {
  try {
    if (process.env.NODE_ENV === "development") {
      return await fetchWeatherData(
        `${local_weathernetServer}/getImsTrueData`,
        cityId
      );
    } else if (process.env.NODE_ENV === "production") {
      return await fetchWeatherData(
        deployed_weathernetServer_getImsTrueData,
        cityId
      );
    }
  } catch (error) {
    console.error(
      "Primary server fetch failed, trying the alternative server:",
      error
    );
    try {
      if (process.env.NODE_ENV === "development") {
        return await fetchWeatherData(
          deployed_weathernetServer_getImsTrueData,
          cityId
        );
      } else if (process.env.NODE_ENV === "production") {
        return await fetchWeatherData(
          `${local_weathernetServer}/getImsTrueData`,
          cityId
        );
      }
    } catch (secondaryError) {
      console.error("Both server fetch attempts failed:", secondaryError);
      throw secondaryError;
    }
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
