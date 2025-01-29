import { useEffect, useState, useCallback } from "react";
import {
  chartSeriesWeatherNet,
  processWeatherNerForecastDataMergeWithImsTrueData,
} from "../../../utils/dataManipulations.js";
import { templateDataOur } from "../../../utils/forecast.js";
import WeatherChart from "../../dataDisplays/weatherChart.js";
import PeriodSlider from "../../dataDisplays/periodSlider.js";
import { ChooseCityAndPeriodBox } from "./weatherForecast.style";
import { getImsTrueData } from "../../../utils/network/weathernetServer";

const WeathernetForecast = () => {
  const [ourDataJson, setOurDataJson] = useState(null);
  const [trueDataJson, setTrueDataJson] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);
  const [chosenTimePeriod, setChosenTimePeriod] = useState([6, 32]);
  const [maxPeriod, setMaxPeriod] = useState(93);

  const fetchData = useCallback(() => {
    setOurDataJson(templateDataOur);
    getImsTrueData(42).then((data) => setTrueDataJson(data)); //TODO: change to city
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    // Process WeatherNet forecast data when dataJson changes
    if (!ourDataJson || !trueDataJson) return;

    const { dataset, minValue, maxValue } =
      processWeatherNerForecastDataMergeWithImsTrueData(
        ourDataJson,
        trueDataJson
      );
    setDataset(dataset);
    setMinValue(minValue);
    setMaxValue(maxValue);
    setMaxPeriod(dataset.length - 1);
  }, [ourDataJson, trueDataJson]);

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
        chartSeries={chartSeriesWeatherNet}
      />
    </>
  );
};

export default WeathernetForecast;
