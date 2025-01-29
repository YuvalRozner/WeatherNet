import React, { useEffect, useState } from "react";
import { processImsForecastData } from "../../../utils/dataManipulations.js";
import { getImsForecast } from "../../../utils/network/weathernetServer";
import ChooseCity from "../../dataDisplays/chooseCity.js";
import PeriodSlider from "../../dataDisplays/periodSlider.js";
import { ChooseCityAndPeriodBox } from "./weatherForecast.style";
import DailyForecast from "../../dataDisplays/dailyForecast.js";
import WeatherTable from "../../dataDisplays/weatherTable.js";
import WeatherChart from "../../dataDisplays/weatherChart.js";
import { chartSeriesIms } from "../../../utils/dataManipulations.js";

const ImsForecast = () => {
  const [dataJsonIms, setDataJsonIms] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [city, setCity] = useState(3); // default city is 3 (Haifa)
  const [chosenTimePeriod, setChosenTimePeriod] = useState([6, 32]);
  const [dailyCountryForecast, setDailyCountryForecast] = useState("");
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);
  const [beginDateForSlider, setBeginDateForSlider] = useState(
    new Date().setHours(0, 0, 0, 0)
  );

  useEffect(() => {
    // Get IMS forecast data when city is changed
    getImsForecast(city).then((data) => setDataJsonIms(data));
  }, [city]);

  useEffect(() => {
    // Process IMS forecast data when dataJson changes
    if (!dataJsonIms) return;

    const { dataset, minValue, maxValue, country } =
      processImsForecastData(dataJsonIms);
    setDataset(dataset);
    setMinValue(minValue);
    setMaxValue(maxValue);
    setDailyCountryForecast(country);
  }, [dataJsonIms]);

  useEffect(() => {
    // Slice dataset based on chosen time period
    if (dataset.length === 0) return;
    const tempSlicedDataset = dataset.slice(
      chosenTimePeriod[0],
      chosenTimePeriod[1]
    );
    setSlicedDataset(tempSlicedDataset);
    setBeginDateForSlider(dataset[0].utcTime);
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
          beginDate={beginDateForSlider}
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
