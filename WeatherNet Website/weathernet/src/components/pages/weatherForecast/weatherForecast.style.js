import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";

export const ChooseCityAndPeriodBox = styled(Box)({
  display: "flex",
  flexDirection: "row",
  alignItems: "center",
  justifyContent: "space-evenly",
  gap: 20,
});

export const MapContainer = styled(Box)({
  display: "flex",
  flexWrap: "wrap",
  alignItems: "center",
  justifyContent: "center",
});
