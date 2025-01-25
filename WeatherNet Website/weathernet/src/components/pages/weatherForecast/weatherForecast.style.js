import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export const ChooseCityAndPeriodBox = styled(Box)({
  display: "flex",
  flexDirection: "row",
  alignItems: "center",
  justifyContent: "space-evenly",
  gap: 20,
});

export const ChartContainerBox = styled(Box)({
  marginLeft: "auto",
  marginRight: "auto",
  width: "90%",
});

export const DayilyForecastText = styled(Typography)({
  marginTop: "10px",
  fontSize: "16px",
  fontWeight: "bold",
});
