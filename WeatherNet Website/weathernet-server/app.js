const express = require("express");
const { spawn, exec } = require("child_process");
const app = express();

app.listen(8080, () => {
  console.log("The WeatherNet server is listening on port 8080");

  // Check the Python version being used
  exec("python --version", (error, stdout, stderr) => {
    if (error) {
      console.error(`Error checking Python version: ${error.message}`);
      return;
    }
    console.log("Python version:", stdout || stderr);
  });
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

app.get("/getImsTrueData", (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");

  const cityId = req.query.cityId; // Get cityId from the query parameters

  const station_id = 42;
  // date =     YYYYMMDDhhmm
  const start_date = "202501260000";
  const end_date = "202501291300";
  const url = `https://ims.gov.il/he/envista_station_all_data_time_range/${station_id}/BP%26DiffR%26Grad%26NIP%26RH%26TD%26TDmax%26TDmin%26TW%26WD%26WDmax%26WS%26WS1mm%26Ws10mm%26Ws10maxEnd%26WSmax%26STDwd%26Rain/${start_date}/${end_date}/1/S`;

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

app.get("/getImsTrue", (req, res) => {
  const python = spawn("python", ["import_and_process_data.py"]);
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
    try {
      const result = JSON.parse(dataToSend);
      res.json({ result });
    } catch (err) {
      console.error("Error parsing JSON from Python script:", err);
      res.status(500).send("Invalid JSON output from Python script");
    }
  });
});
