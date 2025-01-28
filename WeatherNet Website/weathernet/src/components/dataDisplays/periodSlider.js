import React from "react";
import Slider from "@mui/material/Slider";

const PeriodSlider = ({ period, setPeriod, minPeriod, beginDate }) => {
  const indexToHour = (value) => {
    const startDate = new Date(beginDate);
    // startDate.setHours(0, 0, 0, 0);
    const date = new Date(startDate.getTime() + value * 60 * 60 * 1000);
    return `${date.getDate().toString().padStart(2, "0")}/${(date.getMonth() + 1).toString().padStart(2, "0")} ${date.getHours().toString().padStart(2, "0")}:00`;
  };

  const handlePeriodSliderChange = (event, newValue, activeThumb) => {
    if (!Array.isArray(newValue)) {
      return;
    }

    if (newValue[1] - newValue[0] < minPeriod) {
      if (activeThumb === 0) {
        const clamped = Math.min(newValue[0], 100 - minPeriod);
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
      max={93}
      sx={{ width: 500 }}
      getAriaLabel={() => "Time Period"}
      value={period}
      onChange={handlePeriodSliderChange}
      disableSwap
    />
  );
};

export default PeriodSlider;
