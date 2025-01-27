import React, { useState } from "react";
import SpeedDial from "@mui/material/SpeedDial";
import SpeedDialAction from "@mui/material/SpeedDialAction";
import { sharingOptions } from "../../utils/SahringList";
import Tooltip from "@mui/material/Tooltip";
import { ShareContainerBox, StyledShareIcon } from "./topBar.style";

export default function ControlledOpenSpeedDial({ shareUrl, title }) {
  const [open, setOpen] = useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  return (
    <ShareContainerBox>
      <Tooltip title="Share This Page" arrow placement="left-start">
        <SpeedDial
          ariaLabel="share options"
          onClose={handleClose}
          onOpen={handleOpen}
          open={open}
          direction="down"
          icon={<StyledShareIcon />}
        >
          {sharingOptions.map(
            ({ name, IconComponent, url, action, windowName }) => (
              <SpeedDialAction
                sx={{
                  margin: "5px",
                  "@media (max-width: 600px)": {
                    margin: "2px",
                  },
                }}
                key={name}
                icon={
                  <IconComponent
                    style={{
                      borderRadius: "50px",
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
                  if (action) {
                    action(shareUrl);
                  } else {
                    const shareWindowOptions = "width=800,height=400";
                    window.open(
                      url(shareUrl, title),
                      windowName,
                      shareWindowOptions
                    );
                  }
                  handleClose(); // Close the SpeedDial when an action is clicked
                }}
              />
            )
          )}
        </SpeedDial>
      </Tooltip>
    </ShareContainerBox>
  );
}
