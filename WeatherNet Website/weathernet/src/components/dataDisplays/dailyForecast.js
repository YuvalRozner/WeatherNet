import React from "react";
import { Typography, Box } from "@mui/material";

const DailyForecast = ({ dailyCountryForecast }) => {
  return (
    <Box sx={{ marginBottom: "38px" }}>
      <Typography variant="h5" sx={{ marginLeft: "10px" }}>
        Daily Country Forecast:
      </Typography>
      {Array.isArray(dailyCountryForecast) ? (
        dailyCountryForecast.map((forecast, index) => (
          <Typography key={index} variant="body1" sx={{ marginTop: "6px" }}>
            <strong>{forecast.date}</strong>: {forecast.description}
          </Typography>
        ))
      ) : (
        <Typography>No forecast data available</Typography>
      )}
    </Box>
  );
};

export default DailyForecast;
