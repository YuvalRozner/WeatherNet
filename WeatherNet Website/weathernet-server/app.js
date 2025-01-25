const express = require("express");
const { spawn } = require("child_process");
const app = express();

app.listen(8080, () => {
  console.log("The WeatherNet server is listening on port 8080");
});

app.get("/", (req, res) => {
  res.send("Welcome to the WeatherNet server !");
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

app.get("/runPython", (req, res) => {
  const python = spawn("python", ["script.py"]);
  let dataToSend = "";

  python.stdout.on("data", (data) => {
    dataToSend += data.toString();
  });

  python.stderr.on("data", (data) => {
    console.error(`Error from Python script: ${data}`);
  });

  python.on("close", (code) => {
    if (code !== 0) {
      res.status(500).send("Python script execution failed");
      return;
    }
    res.json({ result: dataToSend });
  });
});
