const { onRequest } = require("firebase-functions/v2/https");
const fetch = require("node-fetch");
const logger = require("firebase-functions/logger");

const region = "europe-west1"; // Define the region to deploy to

exports.getImsForecast = onRequest({ region }, (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");

  const cityId = req.query.cityId;

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
      logger.error("Error fetching data from IMS:", error);
      res.status(500).send("Error fetching data from IMS");
    });
});

exports.getImsTrueData = onRequest({ region }, (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");

  const station_id = req.query.cityId; // Get cityId from the query parameters; // 42 is the station id for Haifa
  // date =     YYYYMMDDhhmm
  const now = new Date();
  const end_date = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
    now.getHours(),
    0,
    0
  );
  const start_date = new Date(end_date.getTime() - 3 * 24 * 60 * 60 * 1000);

  const formatDate = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${year}${month}${day}${hours}${minutes}`;
  };

  // const formatted_start_date = formatDate(start_date);
  const formatted_end_date = formatDate(end_date);
  const formatted_start_date = "202501270000";
  const url = `https://ims.gov.il/he/envista_station_all_data_time_range/${station_id}/BP%26DiffR%26Grad%26NIP%26RH%26TD%26TDmax%26TDmin%26TW%26WD%26WDmax%26WS%26WS1mm%26Ws10mm%26Ws10maxEnd%26WSmax%26STDwd%26Rain/${formatted_start_date}/${formatted_end_date}/1/S`;

  fetch(url)
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
      logger.error("Error fetching data from IMS:", error);
      res.status(500).send("Error fetching data from IMS");
    });
});
