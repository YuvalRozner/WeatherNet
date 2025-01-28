import { WeatherNetLogoLayout } from "../../base/weathenetLogoLayout.js";
import ComparingChart from "../weatherForecast/comparingChart.js";

export function Home() {
  return (
    <>
      <ComparingChart />
      <WeatherNetLogoLayout />
    </>
  );
}

export default Home;
