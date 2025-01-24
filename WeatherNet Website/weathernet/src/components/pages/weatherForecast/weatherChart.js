import React from "react";
import { LineChart } from "@mui/x-charts/LineChart";
import { Box } from "@mui/material";

const WeatherChart = ({
  dataset,
  formattedXAxis,
  formattedYAxis,
  forecastChartSeries,
}) => {
  return (
    <Box>
      <LineChart
        loading={dataset.length === 0}
        dataset={dataset}
        xAxis={formattedXAxis}
        yAxis={formattedYAxis}
        series={forecastChartSeries}
        height={400}
        margin={{ left: 60, right: 30, top: 30, bottom: 50 }}
        grid={{ vertical: true, horizontal: true }}
      />
    </Box>
  );
};

export default WeatherChart;
