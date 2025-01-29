import React, { useEffect, useState, useCallback } from "react";
import { processImsForecastDataMergeWithTrueData } from "../../../utils/dataManipulations.js";
import {
  getImsForecast,
  getImsTrueData,
} from "../../../utils/network/gateway.js";
import ChooseCity from "../../dataDisplays/chooseCity.js";
import PeriodSlider from "../../dataDisplays/periodSlider.js";
import { ChooseCityAndPeriodBox } from "./weatherForecast.style";
import DailyForecast from "../../dataDisplays/dailyForecast.js";
import WeatherTable from "../../dataDisplays/weatherTable.js";
import WeatherChart from "../../dataDisplays/weatherChart.js";
import { chartSeriesIms } from "../../../utils/dataManipulations.js";

const ImsForecast = () => {
  const [forecastDataJson, setForecastDataJson] = useState(null);
  const [trueDataJson, setTrueDataJson] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [city, setCity] = useState(3); // default city is 3 (Haifa)
  const [chosenTimePeriod, setChosenTimePeriod] = useState([56, 96]);
  const [maxPeriod, setMaxPeriod] = useState(93);
  const [dailyCountryForecast, setDailyCountryForecast] = useState("");
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);

  const fetchForecastData = useCallback(() => {
    getImsForecast(city).then((data) => setForecastDataJson(data));
    getImsTrueData(42).then((data) => setTrueDataJson(data)); //TODO: change to city
  }, [city]);

  useEffect(() => {
    fetchForecastData();
  }, [fetchForecastData]);

  useEffect(() => {
    if (!forecastDataJson || !trueDataJson) return;

    const { dataset, minValue, maxValue, country } =
      processImsForecastDataMergeWithTrueData(forecastDataJson, trueDataJson);
    setDataset(dataset);
    setMinValue(minValue);
    setMaxValue(maxValue);
    setDailyCountryForecast(country);
    setMaxPeriod(dataset.length - 1);
  }, [forecastDataJson, trueDataJson]);

  useEffect(() => {
    if (dataset.length === 0) return;
    const tempSlicedDataset = dataset.slice(
      chosenTimePeriod[0],
      chosenTimePeriod[1] + 1
    );
    setSlicedDataset(tempSlicedDataset);
  }, [dataset, chosenTimePeriod]);

  return (
    <>
      <DailyForecast dailyCountryForecast={dailyCountryForecast} />
      <ChooseCityAndPeriodBox>
        <ChooseCity setCity={setCity} />
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
        chartSeries={chartSeriesIms}
      />
      <WeatherTable dataset={slicedDataset} />
    </>
  );
};

export default ImsForecast;
