import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export const ChooseCityAndPeriodBox = styled(Box)({
  display: "flex",
  flexDirection: "row",
  alignItems: "center",
  gap: 20,
});

export const ChartContainerBox = styled(Box)({
  marginTop: "20px",
});

export const DayilyForecastText = styled(Typography)({
  marginTop: "10px",
  fontSize: "16px",
  fontWeight: "bold",
});
