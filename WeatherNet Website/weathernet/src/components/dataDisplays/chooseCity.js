import React from "react";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import imsStations from "../../utils/staticData/imsStations";

const ChooseCity = ({ setCity }) => {
  return (
    <Autocomplete
      disablePortal
      options={imsStations}
      sx={{ width: 300 }}
      defaultValue={imsStations[2]}
      onChange={(event, newValue) => setCity(newValue.id)}
      renderInput={(params) => <TextField {...params} label="City" />}
    />
  );
};

export default ChooseCity;
