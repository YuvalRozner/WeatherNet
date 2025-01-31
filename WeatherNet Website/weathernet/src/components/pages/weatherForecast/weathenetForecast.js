import { useEffect, useState, useCallback } from "react";
import {
  chartSeriesWeatherNet,
  processWeatherNerForecastDataMergeWithImsTrueData,
} from "../../../utils/dataManipulations.js";
import { templateDataOur } from "../../../utils/forecast.js";
import WeatherChart from "../../dataDisplays/weatherChart.js";
import PeriodSlider from "../../dataDisplays/periodSlider.js";
import { ChooseCityAndPeriodBox, MapContainer } from "./weatherForecast.style";
import { getImsTrueData } from "../../../utils/network/gateway.js";
import ImageDialog from "../../dataDisplays/imageDialog.js";
import { Skeleton, Tooltip, Typography, Box } from "@mui/material";

const WeathernetForecast = () => {
  const [ourDataJson, setOurDataJson] = useState(null);
  const [trueDataJson, setTrueDataJson] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [slicedDataset, setSlicedDataset] = useState([]);
  const [minValue, setMinValue] = useState(null);
  const [maxValue, setMaxValue] = useState(null);
  const [chosenTimePeriod, setChosenTimePeriod] = useState([56, 96]);
  const [maxPeriod, setMaxPeriod] = useState(93);
  const [mapImageLoaded, setMapImageLoaded] = useState(false);
  const [dialogMapOpen, setDialogMapOpen] = useState(false);

  const handleDialogMapOpen = () => {
    setDialogMapOpen(true);
  };

  const handleDialogMapClose = () => {
    setDialogMapOpen(false);
  };

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
      <MapContainer>
        {!mapImageLoaded && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "row",
              gap: "1rem",
              width: "100%",
            }}
          >
            <Skeleton
              variant="rectangular"
              width="46%"
              height={400}
              sx={{ marginRight: "1rem", borderRadius: "10px" }}
            />
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: "1rem",
                width: "90%",
                marginLeft: "1rem",
                marginBottom: "3rem",
              }}
            >
              <Skeleton
                variant="rectangular"
                width="40%"
                height={32}
                sx={{ borderRadius: "6px" }}
              />
              <Skeleton
                variant="rectangular"
                width="50%"
                height={22}
                sx={{ borderRadius: "5px" }}
              />
              <Skeleton
                variant="rectangular"
                width="50%"
                height={22}
                sx={{ borderRadius: "5px" }}
              />
            </Box>
          </Box>
        )}
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <Tooltip title="Click to open wider" arrow>
            <img
              src="/figures/erea_forecast_ims-station_with_table.png"
              alt="WeatherNet Architecture"
              style={{
                maxWidth: "40%",
                height: "auto",
                display: mapImageLoaded ? "block" : "none",
                cursor: "pointer",
              }}
              onLoad={() => setMapImageLoaded(true)}
              onClick={handleDialogMapOpen}
            />
          </Tooltip>
          {mapImageLoaded && (
            <Typography
              variant="h4"
              sx={{ marginLeft: "2.5rem", marginBottom: "5rem" }}
            >
              Area of the forecasted weather
              <Typography sx={{ fontSize: "1.2rem", marginTop: "0.5rem" }}>
                This Image shows the area of the forecasted weather and the
                IMS's measurement stations used for this forecast.
              </Typography>
            </Typography>
          )}
        </Box>
      </MapContainer>
      <ImageDialog
        open={dialogMapOpen}
        handleClose={handleDialogMapClose}
        image="/figures/erea_forecast_ims-station_with_table.png"
        title="Area of the forecasted weather"
        description="This Image shows the area of the forecasted weather and the IMS's measurement stations used for this forecast."
      />
    </>
  );
};

export default WeathernetForecast;
