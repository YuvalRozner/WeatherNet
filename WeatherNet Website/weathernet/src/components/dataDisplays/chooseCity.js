import React from "react";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import imsPlaces from "../../utils/staticData/imsStationsMergedForecastWithTrue.js";

const ChooseCity = ({ setCity }) => {
  return (
    <Autocomplete
      disablePortal
      options={imsPlaces}
      sx={{ width: 300 }}
      defaultValue={imsPlaces[2]}
      onChange={(event, newValue) =>
        setCity([newValue.idForecast, newValue.idTrueData])
      }
      renderInput={(params) => <TextField {...params} label="City" />}
    />
  );
};

export default ChooseCity;
