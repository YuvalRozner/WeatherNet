import React from "react";
import { WeatherNetLogoLayout } from "../../base/weathenetLogoLayout.js";
import ModelMetrics from "../statistics/modelMetrics.js";
import ComparingChart from "../weatherForecast/comparingChart.js";
import { Divider } from "@mui/material";

export function Home() {
  return (
    <>
      <ModelMetrics />
      <Divider textAlign="left" variant="middle">
        <b style={{ fontSize: "1.6em" }}>Weather Forecast</b>
      </Divider>
      <ComparingChart />
      <WeatherNetLogoLayout />
    </>
  );
}

export default Home;
