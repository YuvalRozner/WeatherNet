import { LineChart } from "@mui/x-charts/LineChart";
import { useMemo } from "react";
import {
  generateFormattedXAxis,
  generateFormattedYAxis,
} from "../../utils/dataManipulations";
import { ChartContainerBox } from "./dataDisplays.style";

const WeatherChart = ({ dataset, minValue, maxValue, chartSeries }) => {
  // Generate formatted X Axis using useMemo
  const formattedXAxis = useMemo(() => generateFormattedXAxis(), []);

  // Generate formatted Y Axis using useMemo
  const formattedYAxis = useMemo(
    () => generateFormattedYAxis(minValue, maxValue),
    [minValue, maxValue]
  );

  // Generate forecast chart series using useMemo
  const forecastChartSeries = useMemo(() => chartSeries(), []);

  return (
    <ChartContainerBox>
      <LineChart
        loading={dataset.length === 0}
        dataset={dataset}
        xAxis={formattedXAxis}
        yAxis={formattedYAxis}
        series={forecastChartSeries}
        height={400}
        margin={{ left: 80, right: 20, top: 30, bottom: 80 }}
        grid={{ vertical: true, horizontal: true }}
        tickLabelStyle={{
          whiteSpace: "pre-line",
          lineHeight: 1.2,
        }}
      />
    </ChartContainerBox>
  );
};

export default WeatherChart;
