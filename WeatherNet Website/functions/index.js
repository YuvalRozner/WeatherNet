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
