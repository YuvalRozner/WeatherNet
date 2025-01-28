export const processImsForecastData = (dataJson) => {
  // If dataJson or forecast_data is missing, safely return empty defaults
  if (!dataJson?.data?.forecast_data) {
    return {
      dataset1: [],
      minValue1: null,
      maxValue1: null,
      country: [],
    };
  }

  const newDataset = [];
  const now = new Date();
  let minValueTemp = 100.0;
  let maxValueTemp = -100.0;
  const countryData = [];

  Object.entries(dataJson.data.forecast_data).forEach(([date, data]) => {
    const formattedDate = new Date(date).toISOString().split("T")[0];
    if (data.country) {
      countryData.push({
        date: formattedDate,
        description: data.country.description,
      });
    }
  });

  Object.entries(dataJson.data.forecast_data).forEach(
    ([date, forecastData]) => {
      const formattedDate = new Date(date).toISOString().split("T")[0];
      const hourlyForecast = forecastData.hourly;

      Object.entries(hourlyForecast).forEach(([hour, forecast]) => {
        const dateTime = new Date(`${formattedDate}T${hour}`);
        const temp = parseFloat(forecast.precise_temperature);
        if (temp < minValueTemp) {
          minValueTemp = temp;
        }
        if (temp > maxValueTemp) {
          maxValueTemp = temp;
        }

        // Check if this timestamp is before or after "now"
        const isPast = dateTime < now;

        newDataset.push({
          utcTime: dateTime.getTime(),
          formattedTime: formatTimeLabel(dateTime.getTime()),
          ImsTemp: isPast ? null : temp,
          // pastTemp: isPast ? temp : null,
          // futureTemp: isPast ? null : temp,
          rain_chance: forecast.rain_chance,
          wave_height: forecast.wave_height,
          relative_humidity: forecast.relative_humidity,
          wind_speed: forecast.wind_speed,
        });
      });
    }
  );

  return {
    dataset: newDataset,
    minValue: Math.floor(minValueTemp - 0.6, 0),
    maxValue: Math.ceil(maxValueTemp + 0.4, 0),
    country: countryData,
  };
};

export const processWeatherNetForecastData = (dataJson) => {
  // If dataJson or forecast_data is missing, safely return empty defaults
  if (!dataJson?.data?.forecast_data) {
    return {
      dataset2: [],
      minValue2: null,
      maxValue2: null,
    };
  }

  const newDataset = [];
  const now = new Date();
  let minValueTemp = 100.0;
  let maxValueTemp = -100.0;

  Object.entries(dataJson.data.forecast_data).forEach(
    ([date, forecastData]) => {
      const formattedDate = new Date(date).toISOString().split("T")[0];
      const hourlyForecast = forecastData.hourly;

      Object.entries(hourlyForecast).forEach(([hour, forecast]) => {
        const dateTime = new Date(`${formattedDate}T${hour}`);
        const temp = parseFloat(forecast.temperature);
        if (temp < minValueTemp) {
          minValueTemp = temp;
        }
        if (temp > maxValueTemp) {
          maxValueTemp = temp;
        }

        // Check if this timestamp is before or after "now"
        const isPast = dateTime < now;

        newDataset.push({
          utcTime: dateTime.getTime(),
          formattedTime: formatTimeLabel(dateTime.getTime()),
          OurTemp: temp,
        });
      });
    }
  );

  return {
    dataset2: newDataset,
    minValue2: Math.floor(minValueTemp - 0.6, 0),
    maxValue2: Math.ceil(maxValueTemp + 0.4, 0),
  };
};

export const processForecastDataMerge = (dataJsonIms, dataJsonWeatherNet) => {
  // Process the two forecasts independently
  const {
    dataset: datasetIms,
    minValue: minValIms,
    maxValue: maxValIms,
    country: countryData,
  } = processImsForecastData(dataJsonIms);

  const {
    dataset2: datasetWn,
    minValue2: minValWn,
    maxValue2: maxValWn,
  } = processWeatherNetForecastData(dataJsonWeatherNet);

  // We'll store merged entries in a map keyed by utcTime
  const mergedMap = {};

  // First, populate from IMS data
  datasetIms.forEach((item) => {
    mergedMap[item.utcTime] = { ...item };
  });

  // Then, merge WeatherNet data
  datasetWn.forEach((item) => {
    // If this utcTime doesn't exist yet, create a new entry
    if (!mergedMap[item.utcTime]) {
      mergedMap[item.utcTime] = {};
    }

    // Destructure to rename the "ImsTemp" field from the second dataset
    // so it won't overwrite the IMS "ImsTemp"
    const { ImsTemp, ...restProps } = item;
    // We'll call it "weatherNetTemp" if needed, or just skip it
    // mergedMap[item.utcTime].weatherNetTemp = ImsTemp;

    // Copy all other properties (like OurTemp, formattedTime, etc.)
    Object.assign(mergedMap[item.utcTime], restProps);
  });

  // Convert merged map back to an array and sort by utcTime
  const mergedDataset = Object.values(mergedMap).sort(
    (a, b) => a.utcTime - b.utcTime
  );

  // From the merged dataset, pick only the desired keys
  const filteredDataset = mergedDataset.map((item) => {
    const allowedKeys = ["utcTime", "formattedTime", "ImsTemp", "OurTemp"];
    const newObj = {};

    allowedKeys.forEach((key) => {
      if (item[key] !== undefined) {
        newObj[key] = item[key];
      }
    });
    return newObj;
  });

  // Compute final min and max from both sets
  const finalMin = Math.min(minValIms ?? 999, minValWn ?? 999);
  const finalMax = Math.max(maxValIms ?? -999, maxValWn ?? -999);

  // Return the merged results
  return {
    dataset: filteredDataset,
    minValue: finalMin,
    maxValue: finalMax,
    country: countryData, // or merge if second data also has country
  };
};

export const formatTimeLabel = (utcTime) => {
  const date = new Date(utcTime);
  const formattedDate = `${date.getDate().toString().padStart(2, "0")}/${(
    date.getMonth() + 1
  )
    .toString()
    .padStart(2, "0")}`;
  const formattedTime = `${date.getHours().toString().padStart(2, "0")}:${date
    .getMinutes()
    .toString()
    .padStart(2, "0")}`;
  return `${formattedDate} ${formattedTime}`;
};

export const chartSeriesMerged = () => [
  {
    id: "pastTemp",
    dataKey: "truePastTemp",
    label: "Today Past (°C)",
    color: "blue",
  },
  {
    id: "ImsTemp",
    dataKey: "ImsTemp",
    label: "IMS's Forecast (°C)",
    color: "#02620f",
  },
  {
    id: "OurTemp",
    dataKey: "OurTemp",
    label: "WeatherNet's Forecast (°C)",
    color: "#02b2af",
  },
];

export const chartSeriesIms = () => [
  {
    id: "pastTemp",
    dataKey: "truePastTemp",
    label: "Today Past (°C)",
    color: "blue",
  },
  {
    id: "ImsTemp",
    dataKey: "ImsTemp",
    label: "IMS's Forecast (°C)",
    color: "#02620f",
  },
];

export const chartSeriesWeatherNet = () => [
  {
    id: "pastTemp",
    dataKey: "truePastTemp",
    label: "Today Past (°C)",
    color: "blue",
  },
  {
    id: "OurTemp",
    dataKey: "OurTemp",
    label: "WeatherNet's Forecast (°C)",
    color: "#02b2af",
  },
];

export const generateFormattedXAxis = () => [
  {
    scaleType: "utc",
    dataKey: "utcTime",
    label: "Time",
    valueFormatter: formatTimeLabel,
  },
];

export const generateFormattedYAxis = (minValue, maxValue) => [
  {
    label: "Temperature (°C)",
    min: minValue,
    max: maxValue,
  },
];
