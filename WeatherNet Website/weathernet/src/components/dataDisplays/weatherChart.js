import { LineChart } from "@mui/x-charts/LineChart";
import { useMemo } from "react";
import {
  generateFormattedXAxis,
  generateFormattedYAxis,
  generateForecastChartSeries,
} from "../../utils/DataManipulations";
import { ChartContainerBox } from "./dataDisplays.style";

const WeatherChart = ({ dataset, minValue, maxValue }) => {
  // Generate formatted X Axis using useMemo
  const formattedXAxis = useMemo(() => generateFormattedXAxis(), []);

  // Generate formatted Y Axis using useMemo
  const formattedYAxis = useMemo(
    () => generateFormattedYAxis(minValue, maxValue),
    [minValue, maxValue]
  );

  // Generate forecast chart series using useMemo
  const forecastChartSeries = useMemo(() => generateForecastChartSeries(), []);

  return (
    <ChartContainerBox>
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
    </ChartContainerBox>
  );
};

export default WeatherChart;
