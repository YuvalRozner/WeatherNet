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
          <Box
            key={index}
            sx={{
              display: "flex",
              marginTop: "6px",
            }}
          >
            <Box
              sx={{
                fontWeight: "bold",
                minWidth: "100px", // Reduced width to bring content closer
                textAlign: "left",
              }}
            >
              <Typography variant="body1">{`${forecast.date}:`}</Typography>
            </Box>
            <Box sx={{ marginLeft: "5px", flex: 1 }}>
              <Typography variant="body1">{forecast.description}</Typography>
            </Box>
          </Box>
        ))
      ) : (
        <Typography>No forecast data available</Typography>
      )}
    </Box>
  );
};

export default DailyForecast;
