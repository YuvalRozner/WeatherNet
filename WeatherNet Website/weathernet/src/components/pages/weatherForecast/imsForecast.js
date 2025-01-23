import * as React from "react";
import { Box } from "@mui/material";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import { LineChart } from "@mui/x-charts/LineChart";
import { useEffect, useState } from "react";
import imsStations from "../../../utils/network/imsStations";
import { getImsForecast } from "../../../utils/network/weathernetServer";
import Slider from "@mui/material/Slider";

export function name() {
  return <div>yuval</div>;
}

const ImsForecast = () => {
  const [dataset, setDataset] = useState([]);
  const [data, setData] = useState(null);
  const [city, setCity] = useState(3);
  const [period, setPeriod] = React.useState([0, 24]);

  useEffect(() => {
    getImsForecast(city).then((data) => setData(data));
  }, [city]);

  // useEffect(() => {
  //   if (!data) return;
  //   const newDataset = [];
  //   Object.entries(data.data.forecast_data).forEach(([date, forecastData]) => {
  //     const formattedDate = new Date(date).toISOString().split("T")[0];
  //     const hourlyForecast = forecastData.hourly;
  //     Object.entries(hourlyForecast).forEach(([hour, forecast]) => {
  //       const dateTime = new Date(`${formattedDate}T${hour}`);
  //       newDataset.push({
  //         pastTemp: parseFloat(forecast.precise_temperature),
  //         futureTemp: parseFloat(forecast.precise_temperature),
  //         utcTime: dateTime.getTime(),
  //       });
  //     });
  //   });
  //   setDataset(newDataset);
  // }, [data]);

  useEffect(() => {
    if (!data) return;

    const newDataset = [];
    const now = new Date();

    Object.entries(data.data.forecast_data).forEach(([date, forecastData]) => {
      const formattedDate = new Date(date).toISOString().split("T")[0];
      const hourlyForecast = forecastData.hourly;

      Object.entries(hourlyForecast).forEach(([hour, forecast]) => {
        const dateTime = new Date(`${formattedDate}T${hour}`);
        const temp = parseFloat(forecast.precise_temperature);

        // Check if this timestamp is before or after "now"
        const isPast = dateTime < now;

        newDataset.push({
          pastTemp: isPast ? temp : null,
          futureTemp: isPast ? null : temp,
          OurTemp: isPast ? null : temp + (Math.random() * 4 - 2),
          utcTime: dateTime.getTime(),
        });
      });
    });

    setDataset(newDataset);
  }, [data]);

  const minPeriod = 6;

  const handleChange2 = (event, newValue, activeThumb) => {
    if (!Array.isArray(newValue)) {
      return;
    }

    if (newValue[1] - newValue[0] < minPeriod) {
      if (activeThumb === 0) {
        const clamped = Math.min(newValue[0], 100 - minPeriod);
        setPeriod([clamped, clamped + minPeriod]);
      } else {
        const clamped = Math.max(newValue[1], minPeriod);
        setPeriod([clamped - minPeriod, clamped]);
      }
    } else {
      setPeriod(newValue);
    }
  };

  function indexToHour(value) {
    const startDate = new Date();
    startDate.setHours(0, 0, 0, 0);
    const date = new Date(startDate.getTime() + value * 60 * 60 * 1000);
    return `${date.getDate().toString().padStart(2, "0")}/${(date.getMonth() + 1).toString().padStart(2, "0")} ${date.getHours().toString().padStart(2, "0")}:00`;
  }

  // Separate data into two arrays: "pastData" and "futureData"
  const filteredData = dataset.slice(period[0], period[1]);
  const nowTime = Date.now();
  const pastData = filteredData.filter((item) => item.utcTime < nowTime);
  const futureData = filteredData.filter((item) => item.utcTime >= nowTime);

  return (
    <>
      <h3>Hourly Forecast</h3>
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          gap: 8,
        }}
      >
        <Autocomplete
          disablePortal
          options={imsStations}
          sx={{ width: 300 }}
          defaultValue={imsStations[2]}
          onChange={(event, newValue) => setCity(newValue.id)}
          renderInput={(params) => <TextField {...params} label="City" />}
        />
        <Slider
          valueLabelDisplay="on"
          valueLabelFormat={(value) => indexToHour(value)}
          max={93}
          sx={{ width: 500 }}
          getAriaLabel={() => "Time Period"}
          value={period}
          onChange={handleChange2}
          disableSwap
        />
      </Box>
      <Box>
        <LineChart
          loading={dataset.length === 0}
          dataset={dataset.slice(period[0], period[1])}
          xAxis={[
            {
              scaleType: "utc",
              dataKey: "utcTime",
              label: "Time",
              valueFormatter: (utcTime) => {
                const date = new Date(utcTime);
                const formattedDate = `${date.getDate().toString().padStart(1, "0")}/${date.getMonth().toString().padStart(1, "0") + 1}`;
                const formattedTime = `${date.getHours().toString().padStart(2, "0")}:${date.getMinutes().toString().padStart(2, "0")}`;
                return `${formattedDate} ${formattedTime}`;
              },
            },
          ]}
          yAxis={[{ label: "Temperature (°C)" }]}
          series={[
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
          ]}
          height={400}
          margin={{ left: 60, right: 30, top: 30, bottom: 50 }}
          grid={{ vertical: true, horizontal: true }}
        />
      </Box>
    </>
  );
};

export default ImsForecast;
