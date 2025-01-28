import React, { useEffect, useState } from "react";
import {
  processImsForecastData,
  processWeatherNetForecastData,
  mergeByUtcTime,
} from "../../../utils/dataManipulations.js";
import { getImsForecast } from "../../../utils/network/weathernetServer.js";
import ChooseCity from "../../dataDisplays/chooseCity.js";
import PeriodSlider from "../../dataDisplays/periodSlider.js";
import { ChooseCityAndPeriodBox } from "./weatherForecast.style.js";
import WeatherChart from "../../dataDisplays/weatherChart.js";
import { templateDataOur } from "../../../utils/forecast.js";

const ComparingChart = () => {
  const [dataJsonIms, setDataJsonIms] = useState(null);
  const [dataJsonOur, setDataJsonOur] = useState(null);
  const [datasetIms, setDatasetIms] = useState([]);
  const [datasetOur, setDatasetOur] = useState([]);
  const [slicedDatasetIms, setSlicedDatasetIms] = useState([]);
  const [city, setCity] = useState(3); // default city is 3 (Haifa)
  const [chosenTimePeriod, setChosenTimePeriod] = useState([6, 32]);
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);
  const [beginDateForSlider, setBeginDateForSlider] = useState(
    new Date().setHours(0, 0, 0, 0)
  );

  useEffect(() => {
    // Get IMS forecast data when city is changed
    getImsForecast(city).then((data) => setDataJsonIms(data));
    setDataJsonOur(templateDataOur);
  }, [city]);

  useEffect(() => {
    // Process IMS forecast data when dataJson changes
    if (!dataJsonIms) return;

    const { dataset, minValue, maxValue, country } =
      processImsForecastData(dataJsonIms);
    setDatasetIms(dataset);
    const { dataset2, minValue2, maxValue2 } =
      processWeatherNetForecastData(dataJsonOur);
    setDatasetOur(dataset2);
    setMinValue(Math.min(minValue, minValue2));
    setMaxValue(Math.max(maxValue, maxValue2));
  }, [dataJsonIms, dataJsonOur]);

  useEffect(() => {
    // Slice dataset based on chosen time period
    if (datasetIms.length === 0) return;
    const tempSlicedDataset = mergeByUtcTime(datasetIms, datasetOur).slice(
      chosenTimePeriod[0],
      chosenTimePeriod[1]
    );
    setSlicedDatasetIms(tempSlicedDataset);
    setBeginDateForSlider(datasetIms[0].utcTime);
  }, [datasetIms, datasetOur, chosenTimePeriod]);

  return (
    <>
      <ChooseCityAndPeriodBox>
        <ChooseCity setCity={setCity} />
        <PeriodSlider
          period={chosenTimePeriod}
          setPeriod={setChosenTimePeriod}
          minPeriod={6}
          beginDate={beginDateForSlider}
        />
      </ChooseCityAndPeriodBox>
      <WeatherChart
        dataset={slicedDatasetIms}
        minValue={minValue}
        maxValue={maxValue}
      />
    </>
  );
};

export default ComparingChart;
