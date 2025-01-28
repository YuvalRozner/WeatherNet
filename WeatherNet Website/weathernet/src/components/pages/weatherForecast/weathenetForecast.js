import WeatherChart from "./weatherChart";
import { useState, useEffect } from "react";
import { templateDataOur } from "../../../utils/forecast.js";
import { processForecastDataWeatherNet } from "../../../utils/DataManipulations";

const WeathernetForecast = () => {
  const [dataJson, setDataJson] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);

  useEffect(() => {
    setDataJson(templateDataOur);
  }, []);

  useEffect(() => {
    // Process WeatherNet forecast data when dataJson changes
    if (!dataJson) return;

    const { dataset, minValue, maxValue } =
      processForecastDataWeatherNet(dataJson);
    setDataset(dataset);
    setSlicedDataset(dataset); // todo: remove this
    setMinValue(minValue);
    setMaxValue(maxValue);
  }, [dataJson]);

  // useEffect(() => {
  //   // Slice dataset based on chosen time period
  //   if (dataset.length === 0) return;
  //   const tempSlicedDataset = dataset.slice(
  //     chosenTimePeriod[0],
  //     chosenTimePeriod[1]
  //   );
  //   setSlicedDataset(tempSlicedDataset);
  // }, [dataset, chosenTimePeriod]);

  return (
    <WeatherChart
      // dataset={slicedDataset}
      dataset={dataset} // todo: remove this
      minValue={minValue}
      maxValue={maxValue}
    />
  );
};

export default WeathernetForecast;
