export const processForecastData = (dataJson) => {
  // If dataJson or forecast_data is missing, safely return empty defaults
  if (!dataJson?.data?.forecast_data) {
    return {
      dataset: [],
      minValue: null,
      maxValue: null,
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
          ImsTemp: temp,
          pastTemp: isPast ? temp : null,
          futureTemp: isPast ? null : temp,
          OurTemp: isPast ? null : temp + (Math.random() * 2 - 1),
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

export const generateForecastChartSeries = () => [
  {
    id: "pastTemp",
    dataKey: "pastTemp",
    label: "Today Past (°C)",
    color: "blue",
  },
  {
    id: "futureTemp",
    dataKey: "futureTemp",
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
