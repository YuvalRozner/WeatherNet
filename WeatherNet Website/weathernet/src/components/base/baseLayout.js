import { useState } from "react";
import { HelpIcon, HelpButtonContainer } from "./baseLayout.style.js";
import { HelpDialog } from "./helpDialog.js";

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
        <HelpIcon />
      </HelpButtonContainer>
      <HelpDialog open={open} onClose={handleClose} />
    </>
  );
}
