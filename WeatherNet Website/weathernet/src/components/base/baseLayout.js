import { useState } from "react";
import { HelpIcon, HelpButtonContainer } from "./baseLayout.style.js";
import { HelpDialog } from "./helpDialog.js";
import Tooltip from "@mui/material/Tooltip";

export function BaseLayout() {
  const [open, setOpen] = useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <>
      <HelpButtonContainer onClick={handleClickOpen}>
        <Tooltip title="Need Help?" arrow placement="top-start">
          <HelpIcon />
        </Tooltip>
      </HelpButtonContainer>
      <HelpDialog open={open} onClose={handleClose} />
    </>
  );
}
