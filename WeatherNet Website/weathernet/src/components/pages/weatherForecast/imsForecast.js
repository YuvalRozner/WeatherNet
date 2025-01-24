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
  const [city, setCity] = useState(3);
  const [chosenTimePeriod, setChosenTimePeriod] = useState([6, 32]);
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);
  const [columns, setColumns] = useState([]);
  const [rows, setRows] = useState([]);
  const [dailyCountryForecast, setDailyCountryForecast] = useState("");

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

  // Generate formatted X Axis using useMemo
  const formattedXAxis = useMemo(() => generateFormattedXAxis(), []);

  // Generate formatted Y Axis using useMemo
  const formattedYAxis = useMemo(
    () => generateFormattedYAxis(minValue, maxValue),
    [minValue, maxValue]
  );

  // Generate forecast chart series using useMemo
  const forecastChartSeries = useMemo(() => generateForecastChartSeries(), []);

  // Build columns & rows for the transposed table
  useEffect(() => {
    if (slicedDataset.length === 0) {
      setColumns([]);
      setRows([]);
      return;
    }

    const newColumns = [
      { id: "parameter", label: "Parameter", minWidth: 170 },
      ...slicedDataset.map((item, index) => ({
        id: `time-${index}`,
        label: item.formattedTime,
        minWidth: 60,
      })),
    ];

    const paramRows = [
      { parameter: "Temperature (°C)", paramKey: "ImsTemp" },
      { parameter: "Rain Chance (%)", paramKey: "rain_chance" },
      { parameter: "Wave Height (m)", paramKey: "wave_height" },
      { parameter: "Relative Humidity (%)", paramKey: "relative_humidity" },
      { parameter: "Wind Speed (km/h)", paramKey: "wind_speed" },
    ];

    const newRows = paramRows.map((pRow) => {
      const rowObj = { parameter: pRow.parameter };
      slicedDataset.forEach((item, idx) => {
        rowObj[`time-${idx}`] = item[pRow.paramKey] ?? "-";
      });
      return rowObj;
    });

    setColumns(newColumns);
    setRows(newRows);
  }, [slicedDataset]);

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
      <ChartContainerBox>
        <WeatherChart
          dataset={slicedDataset}
          formattedXAxis={formattedXAxis}
          formattedYAxis={formattedYAxis}
          forecastChartSeries={forecastChartSeries}
        />
      </ChartContainerBox>
      <WeatherTable columns={columns} rows={rows} />
    </>
  );
};

export default ImsForecast;
