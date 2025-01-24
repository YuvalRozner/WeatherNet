const express = require("express");
const app = express();

app.listen(8080, () => {
  console.log("The WeatherNet server is listening on port 8080");
});

app.get("/imsForecast", (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");

  const cityId = req.query.cityId; // Get cityId from the query parameters

  fetch(`https://ims.gov.il/en/city_portal/${cityId}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      res.json(data);
    })
    .catch((error) => {
      console.error("Error fetching data from IMS:", error);
      res.status(500).send("Error fetching data from IMS");
    });
});
