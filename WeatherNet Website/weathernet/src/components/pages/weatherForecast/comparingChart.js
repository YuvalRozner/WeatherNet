import React, { useEffect, useState, useCallback } from "react";
import { processForecastfromBothWithTrueMerge } from "../../../utils/dataManipulations.js";
import {
  getImsForecast,
  getImsTrueData,
} from "../../../utils/network/gateway.js";
import PeriodSlider from "../../dataDisplays/periodSlider.js";
import { ChooseCityAndPeriodBox } from "./weatherForecast.style.js";
import WeatherChart from "../../dataDisplays/weatherChart.js";
import { templateDataOur } from "../../../utils/forecast.js";
import { chartSeriesMerged } from "../../../utils/dataManipulations.js";

const ComparingChart = () => {
  const [dataJsonIms, setDataJsonIms] = useState(null);
  const [dataJsonOur, setDataJsonOur] = useState(null);
  const [dataJsonTrue, setDataJsonTrue] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [chosenTimePeriod, setChosenTimePeriod] = useState([51, 114]);
  const [maxPeriod, setMaxPeriod] = useState(93);
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);

  const fetchData = useCallback(() => {
    // Get IMS forecast data when city is changed
    getImsForecast(3).then((data) => setDataJsonIms(data)); //TODO: change to city
    getImsTrueData(42).then((data) => setDataJsonTrue(data)); //TODO: change to city
    setDataJsonOur(templateDataOur);
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    // Process IMS forecast data when dataJson changes
    if (!dataJsonIms) return;

    const { dataset, minValue, maxValue } =
      processForecastfromBothWithTrueMerge(
        dataJsonIms,
        dataJsonOur,
        dataJsonTrue
      );
    setDataset(dataset);
    setMinValue(minValue);
    setMaxValue(maxValue);
    setMaxPeriod(dataset.length - 1);
  }, [dataJsonIms, dataJsonOur, dataJsonTrue]);

  useEffect(() => {
    // Slice dataset based on chosen time period
    if (dataset.length === 0) return;
    const tempSlicedDataset = dataset.slice(
      chosenTimePeriod[0],
      chosenTimePeriod[1] + 1
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
          maxPeriod={maxPeriod}
          dataset={dataset}
        />
      </ChooseCityAndPeriodBox>
      <WeatherChart
        dataset={slicedDataset}
        minValue={minValue}
        maxValue={maxValue}
        chartSeries={chartSeriesMerged}
      />
    </>
  );
};

export default ComparingChart;
