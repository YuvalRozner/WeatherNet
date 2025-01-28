import { useEffect, useState } from "react";
import {
  chartSeriesWeatherNet,
  processWeatherNetForecastData,
} from "../../../utils/dataManipulations.js";
import { templateDataOur } from "../../../utils/forecast.js";
import WeatherChart from "../../dataDisplays/weatherChart.js";
import PeriodSlider from "../../dataDisplays/periodSlider.js";
import { ChooseCityAndPeriodBox } from "./weatherForecast.style";

const WeathernetForecast = () => {
  const [dataJson, setDataJson] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);
  const [chosenTimePeriod, setChosenTimePeriod] = useState([6, 32]);
  const [beginDateForSlider, setBeginDateForSlider] = useState(
    new Date().setHours(0, 0, 0, 0)
  );

  useEffect(() => {
    setDataJson(templateDataOur);
  }, []);

  useEffect(() => {
    // Process WeatherNet forecast data when dataJson changes
    if (!dataJson) return;

    const { dataset2, minValue2, maxValue2 } =
      processWeatherNetForecastData(dataJson);
    setDataset(dataset2);
    setMinValue(minValue2);
    setMaxValue(maxValue2);
  }, [dataJson]);

  useEffect(() => {
    // Slice dataset based on chosen time period
    if (dataset.length === 0) return;
    const tempSlicedDataset = dataset.slice(
      chosenTimePeriod[0],
      chosenTimePeriod[1]
    );
    setSlicedDataset(tempSlicedDataset);
  }, [dataset, chosenTimePeriod]);

  return (
    <>
      <ChooseCityAndPeriodBox>
        <PeriodSlider
          period={chosenTimePeriod}
          setPeriod={setChosenTimePeriod}
          minPeriod={6}
          beginDate={beginDateForSlider}
        />
      </ChooseCityAndPeriodBox>
      <WeatherChart
        dataset={slicedDataset}
        minValue={minValue}
        maxValue={maxValue}
        chartSeries={chartSeriesWeatherNet}
      />
    </>
  );
};

export default WeathernetForecast;
