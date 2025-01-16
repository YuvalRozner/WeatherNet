import React, { useState } from "react";
import Box from "@mui/material/Box";
import SpeedDial from "@mui/material/SpeedDial";
import SpeedDialAction from "@mui/material/SpeedDialAction";
import ShareIcon from "@mui/icons-material/Share";
import { sharingOptions } from "../../utils/SahringList";

export default function ControlledOpenSpeedDial({ shareUrl, title }) {
  const [open, setOpen] = useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  return (
    <Box sx={{ position: "absolute", top: "11px", right: "30px" }}>
      <SpeedDial
        ariaLabel="share options"
        onClose={handleClose}
        onOpen={handleOpen}
        open={open}
        direction="down"
        icon={<ShareIcon style={{ width: "24px", height: "24px" }} />}
        FabProps={{ size: "small" }}
      >
        {sharingOptions.map(({ name, IconComponent, url, windowName }) => (
          <SpeedDialAction
            key={name}
            icon={
              <IconComponent
                style={{
                  borderRadius: "45px",
                  width: "100%",
                  height: "100%",
                  transition: "transform 0.3s ease",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                }}
              />
            }
            tooltipTitle={name}
            onClick={() => {
              const shareWindowOptions = "width=800,height=400";
              window.open(url(shareUrl, title), windowName, shareWindowOptions);
              handleClose(); // Close the SpeedDial when an action is clicked
            }}
          />
        ))}
      </SpeedDial>
    </Box>
  );
}
