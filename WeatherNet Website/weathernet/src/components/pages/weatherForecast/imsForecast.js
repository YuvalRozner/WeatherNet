import { useEffect, useState } from "react";
import { Box } from "@mui/material";
import { ResponsiveChartContainer } from "@mui/x-charts/ResponsiveChartContainer";
import { LinePlot, MarkPlot } from "@mui/x-charts/LineChart";
import { ChartsXAxis } from "@mui/x-charts/ChartsXAxis";
import { ChartsYAxis } from "@mui/x-charts/ChartsYAxis";
import { ChartsGrid } from "@mui/x-charts/ChartsGrid";
import { ChartsTooltip } from "@mui/x-charts/ChartsTooltip";
import { fetchWeatherData } from "../../../utils/network/weathernetServer";

const ImsForecast = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchWeatherData(3).then((data) => setData(data));
  }, []);

  if (!data || !data.data) {
    return <div>Loading...</div>;
  }

  const forecast = data.data.forecast_data["2025-01-23"].hourly;
  // const forecast = data.data.fixed_forecast_data["2025-01-23"].hourly;
  const dataset = Object.keys(forecast).map((hour) => ({
    temp: parseFloat(forecast[hour].precise_temperature),
    time: hour,
  }));

  return (
    <>
      <h3>Hourly Forecast</h3>
      <Box sx={{ width: "100%" }}>
        <ResponsiveChartContainer
          series={[{ type: "line", dataKey: "temp" }]}
          xAxis={[
            {
              scaleType: "band",
              dataKey: "time",
              label: "Time",
            },
          ]}
          yAxis={[{ id: "leftAxis" }]}
          dataset={dataset}
          height={400}
        >
          <ChartsGrid horizontal />
          <LinePlot />
          <MarkPlot />

          <ChartsXAxis />
          <ChartsYAxis axisId="leftAxis" label="temerature (°C)" />
          <ChartsTooltip />
        </ResponsiveChartContainer>
      </Box>
    </>
  );
};

export default ImsForecast;
