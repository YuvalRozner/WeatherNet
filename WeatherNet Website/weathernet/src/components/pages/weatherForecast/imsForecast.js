import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@mui/material";
import TableContainer from "@mui/material/TableContainer";
import React, { useEffect, useMemo, useState } from "react";
import {
  generateForecastChartSeries,
  generateFormattedXAxis,
  generateFormattedYAxis,
  processForecastData,
  formatTimeLabel,
} from "../../../utils/DataManipulations";
import { getImsForecast } from "../../../utils/network/weathernetServer";
import ChooseCity from "./chooseCity";
import PeriodSlider from "./periodSlider";
import {
  ChartContainerBox,
  ChooseCityAndPeriodBox,
} from "./weatherForecast.style";
import DailyForecast from "./dailyForecast";
import WeatherTable from "./weatherTable";
import WeatherChart from "./weatherChart";

const ImsForecast = () => {
  const [dataJson, setDataJson] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [city, setCity] = useState(3); // default city is 3 (Haifa)
  const [chosenTimePeriod, setChosenTimePeriod] = useState([6, 32]);
  const [dailyCountryForecast, setDailyCountryForecast] = useState("");
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);

  useEffect(() => {
    // Get IMS forecast data when city is changed
    getImsForecast(city).then((data) => setDataJson(data));
  }, [city]);

  useEffect(() => {
    // Process IMS forecast data when dataJson changes
    if (!dataJson) return;

    const { dataset, minValue, maxValue, country } =
      processForecastData(dataJson);
    setDataset(dataset);
    setMinValue(minValue);
    setMaxValue(maxValue);
    setDailyCountryForecast(country);
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
      <DailyForecast dailyCountryForecast={dailyCountryForecast} />
      <ChooseCityAndPeriodBox>
        <ChooseCity setCity={setCity} />
        <PeriodSlider
          period={chosenTimePeriod}
          setPeriod={setChosenTimePeriod}
          minPeriod={6}
        />
      </ChooseCityAndPeriodBox>
      <WeatherChart
        dataset={slicedDataset}
        minValue={minValue}
        maxValue={maxValue}
      />
      <WeatherTable dataset={slicedDataset} />
    </>
  );
};

export default ImsForecast;
