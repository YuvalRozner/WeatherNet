import React, { useState } from "react";
import IconButton from "@mui/material/IconButton";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import SettingsIcon from "@mui/icons-material/Settings";
import Button from "@mui/material/Button";
import { useNotifications } from "@toolpad/core/useNotifications";
import UploadIcon from "@mui/icons-material/Upload";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

const Settings = () => {
  const [anchorEl, setAnchorEl] = useState(null);
  const open = Boolean(anchorEl);
  const notifications = useNotifications();

  const handleSettingsClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleSettingsClose = () => {
    setAnchorEl(null);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/json") {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target.result);
          // Save the uploaded JSON to sessionStorage
          sessionStorage.setItem("uploadedForecast", JSON.stringify(json));
          notifications.show("Forecast JSON uploaded successfully !", {
            severity: "success",
            autoHideDuration: 3000,
          });
          handleSettingsClose(); // Close the menu after successful upload
        } catch (error) {
          console.error("Invalid JSON file");
          notifications.show("Failed to upload: Invalid JSON file.", {
            severity: "error",
            autoHideDuration: 3000,
          });
        }
      };
      reader.readAsText(file);
    } else {
      console.error("Please upload a valid JSON file.");
      notifications.show("Failed to upload: Please select a valid JSON file.", {
        severity: "error",
        autoHideDuration: 3000,
      });
    }
  };

  const handleOpenColabNotebook = () => {
    window.open(
      "https://colab.research.google.com/github/YuvalRozner/WeatherNet/blob/main/Backend/Model_Pytorch/utils/inference.ipynb",
      "_blank"
    );
    handleSettingsClose();
  };

  return (
    <>
      <IconButton color="inherit" onClick={handleSettingsClick}>
        <SettingsIcon style={{ color: "#58a6ff" }} />
      </IconButton>
      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleSettingsClose}
        onMouseLeave={handleSettingsClose} // Close menu when mouse leaves
        anchorOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
        MenuListProps={{
          sx: {
            "& .MuiMenuItem-root": {
              "&:hover": {
                backgroundColor: "transparent",
              },
            },
          },
        }}
      >
        <MenuItem>
          <Button
            variant="outlined"
            component="label"
            startIcon={<UploadIcon />}
            sx={{
              color: "#fff",
              "&:hover": {
                backgroundColor: "rgba(41, 184, 200, 0.7)",
              },
              padding: "8px 16px",
              borderRadius: "8px",
              textTransform: "none",
              fontSize: "1rem",
            }}
          >
            Upload Forecast JSON
            <input
              type="file"
              accept=".json"
              hidden
              onChange={handleFileUpload}
            />
          </Button>
        </MenuItem>
        <MenuItem>
          <Button
            variant="outlined"
            startIcon={<OpenInNewIcon />}
            onClick={handleOpenColabNotebook}
            sx={{
              color: "#fff",
              "&:hover": {
                backgroundColor: "rgba(41, 184, 200, 0.7)",
              },
              padding: "8px 16px",
              borderRadius: "8px",
              textTransform: "none",
              fontSize: "1rem",
            }}
          >
            Open Colab Notebook
          </Button>
        </MenuItem>
      </Menu>
    </>
  );
};

export default Settings;
