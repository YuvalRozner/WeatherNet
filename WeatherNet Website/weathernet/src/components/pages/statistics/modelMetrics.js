import React from "react";
import { getmodelMetricsData } from "../../../utils/staticData/statisticsData";
import { Tooltip } from "@mui/material";
import {
  CardsContainer,
  MetricCard,
  CardHeader,
  MetricTitle,
  OverallValue,
  ChartWrapper,
  StyledBarChart,
} from "./statistics.style";
import { MyTheme } from "../../../utils/theme";

export default function ModelMetrics() {
  const metrics = getmodelMetricsData();

  // Cycle through a set of border colors for each card:
  const borderColors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#ff8c00"];

  const chartsWidth =
    window.innerWidth <= MyTheme.breakpoints.values.xl ? 210 : 235;
  const chartsHeight =
    window.innerWidth <= MyTheme.breakpoints.values.xl ? 90 : 80;

  return (
    <CardsContainer>
      {metrics.map((item, i) => {
        // subIntervals are the keys in item.models: ["1-12", "13-24", ...]
        const subIntervals = Object.keys(item.models);
        // subValues are the numeric values
        const subValues = Object.values(item.models);

        // Calculate max value for current metric to scale y-axis
        const metMax = Math.max(...subValues, 0.0001);

        return (
          <Tooltip key={item.type} title={item.explanation || ""}>
            <MetricCard borderColor={borderColors[i % borderColors.length]}>
              {/* Left side: metric type; right side: overall value */}
              <CardHeader>
                <MetricTitle>{item.type}</MetricTitle>
                <OverallValue>{item.overall.toFixed(1)}</OverallValue>
              </CardHeader>

              <ChartWrapper>
                <StyledBarChart
                  borderColor={borderColors[i % borderColors.length]}
                  xAxis={[
                    {
                      data: subIntervals,
                      scaleType: "band",
                    },
                  ]}
                  series={[
                    {
                      data: subValues,
                      label: item.type,
                      barSize: 10,
                      dataLabel: {
                        visible: true,
                        formatter: (val) => val.toFixed(1),
                        anchor: "start", // Position label above the bar
                        offset: 8,
                      },
                    },
                  ]}
                  width={chartsWidth} /* match card width */
                  height={chartsHeight} /* adjusted height for better fit */
                  margin={{
                    top: 12,
                    bottom: 20,
                    left: 25,
                    right: 20,
                  }} /* adjusted top margin */
                  legend={{ hidden: true }}
                  yAxis={[{ min: 0, max: Math.ceil(metMax) }]}
                />
              </ChartWrapper>
            </MetricCard>
          </Tooltip>
        );
      })}
    </CardsContainer>
  );
}
