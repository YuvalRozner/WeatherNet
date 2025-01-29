import React from "react";
import Slider from "@mui/material/Slider";

const PeriodSlider = ({ period, setPeriod, minPeriod, maxPeriod, dataset }) => {
  const indexToHour = (value) => {
    if (dataset.length === 0) return "";
    const date = new Date(dataset[value].utcTime);
    return `${date.getDate().toString().padStart(2, "0")}/${(date.getMonth() + 1).toString().padStart(2, "0")} ${date.getHours().toString().padStart(2, "0")}:00`;
  };

  const handlePeriodSliderChange = (event, newValue, activeThumb) => {
    if (!Array.isArray(newValue)) {
      return;
    }

    if (newValue[1] - newValue[0] < minPeriod) {
      if (activeThumb === 0) {
        const clamped = Math.min(newValue[0], maxPeriod - minPeriod);
        setPeriod([clamped, clamped + minPeriod]);
      } else {
        const clamped = Math.max(newValue[1], minPeriod);
        setPeriod([clamped - minPeriod, clamped]);
      }
    } else {
      setPeriod(newValue);
    }
  };

  return (
    <Slider
      valueLabelDisplay="on"
      valueLabelFormat={(value) => indexToHour(value)}
      max={maxPeriod}
      sx={{ width: 500 }}
      getAriaLabel={() => "Time Period"}
      value={period}
      onChange={handlePeriodSliderChange}
      disableSwap
    />
  );
};

export default PeriodSlider;
