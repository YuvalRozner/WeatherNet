import React, { useState } from "react";
import { WeatherNetLogoLayout } from "../../base/weathenetLogoLayout.js";
import ModelMetrics from "../statistics/modelMetrics.js";
import ComparingChart from "../weatherForecast/comparingChart.js";
import { Divider, Skeleton, Tooltip, useTheme } from "@mui/material";
import { ForecastContainer, ArchitectureContainer } from "./home.style.js";
import ImageDialog from "../../dataDisplays/imageDialog.js";
import { Typography } from "@mui/material";

export function Home() {
  const [mapImageLoaded, setMapImageLoaded] = useState(false);
  const [dialogMapOpen, setDialogMapOpen] = useState(false);
  const [architectureImageLoaded, setArchitectureImageLoaded] = useState(false);
  const [dialogArchitectureOpen, setDialogArchitectureOpen] = useState(false);

  const theme = useTheme();
  const themeMode = theme.palette.mode;

  const handleDialogMapOpen = () => {
    setDialogMapOpen(true);
  };

  const handleDialogMapClose = () => {
    setDialogMapOpen(false);
  };

  const handleDialogArchitectureOpen = () => {
    setDialogArchitectureOpen(true);
  };

  const handleDialogArchitectureClose = () => {
    setDialogArchitectureOpen(false);
  };

  return (
    <>
      <ModelMetrics />
      <Divider
        textAlign="left"
        variant="middle"
        sx={{ margin: "1.2rem 0 1rem 0" }}
      >
        <b style={{ fontSize: "1.6em" }}>Weather Forecast</b>
      </Divider>
      <div style={{ fontSize: "1.2rem", marginBottom: "2rem" }}>
        You are now viewing the forecasts made on the 31/01/2025 15:00 and on
        the 03/02/2025 01:00.
        <br />
        If you want to view the forecasts made on the current date, please use
        the colab inference notbook via the settings and upload the forecast
        json file.
      </div>

      <ForecastContainer
        style={{ display: "flex", flexDirection: "row", alignItems: "center" }}
      >
        {!mapImageLoaded && (
          <Skeleton
            variant="rectangular"
            width="30%"
            height={200}
            sx={{ marginRight: "1rem", borderRadius: "10px" }}
          />
        )}
        <Tooltip title="Click to open wider" arrow>
          <img
            src="/figures/erea_forecast_ims-station_with_table.png"
            alt="WeatherNet Architecture"
            style={{
              maxWidth: "350px",
              height: "auto",
              display: mapImageLoaded ? "block" : "none",
              cursor: "pointer",
            }}
            onLoad={() => setMapImageLoaded(true)}
            onClick={handleDialogMapOpen}
          />
        </Tooltip>
        <div style={{ flex: 1 }}>
          <ComparingChart />
        </div>
      </ForecastContainer>

      <ImageDialog
        open={dialogMapOpen}
        handleClose={handleDialogMapClose}
        image="/figures/erea_forecast_ims-station_with_table.png"
        title="Area of the forecasted weather"
        description="This Image shows the area of the forecasted weather and the IMS's measurement stations used for this forecast."
      />

      <Divider
        textAlign="left"
        variant="middle"
        sx={{ margin: "1.2rem 0 2.7rem 0" }}
      >
        <b style={{ fontSize: "1.6em" }}>Our Architecture</b>
      </Divider>
      <ArchitectureContainer
        style={{ display: "flex", flexDirection: "row", alignItems: "center" }}
      >
        {!architectureImageLoaded && (
          <Skeleton
            variant="rectangular"
            width="70%"
            height={200}
            sx={{ marginRight: "1rem", borderRadius: "10px" }}
          />
        )}
        <Tooltip title="Click to open wider" arrow>
          <img
            src={`/figures/architecture_${themeMode}.png`}
            alt="WeatherNet Architecture"
            style={{
              maxWidth: "70%",
              height: "auto",
              display: architectureImageLoaded ? "block" : "none",
              cursor: "pointer",
            }}
            onLoad={() => setArchitectureImageLoaded(true)}
            onClick={handleDialogArchitectureOpen}
          />
        </Tooltip>
        <WeatherNetLogoLayout />
      </ArchitectureContainer>

      <ImageDialog
        open={dialogArchitectureOpen}
        handleClose={handleDialogArchitectureClose}
        image={`/figures/architecture_${themeMode}.png`}
        title="WeatherNet Architecture"
        description={`This Image shows the architecture of the WeatherNet model.\nLearn more about the architecture in the 'Architecture' page.`}
      />
    </>
  );
}

export default Home;
