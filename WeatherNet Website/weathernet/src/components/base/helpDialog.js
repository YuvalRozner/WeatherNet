import React from "react";
import { Dialog, DialogTitle, DialogContent, IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { DraggablePaperComponent } from "../../utils/draggable.js";

export function HelpDialog({ open, onClose }) {
  return (
    <>
      <Dialog
        open={open}
        onClose={onClose}
        maxWidth="md"
        fullWidth
        PaperComponent={DraggablePaperComponent}
      >
        <DialogTitle id="draggable-dialog-title" style={{ cursor: "move" }}>
          User Manual
          <IconButton
            aria-label="close"
            onClick={onClose}
            sx={{
              position: "absolute",
              right: 8,
              top: 8,
              color: (theme) => theme.palette.grey[500],
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          <iframe
            src="/papers/WeatherNet - User Manual.pdf"
            width="100%"
            height="600px"
            title="WeatherNet - User Manual"
          />
        </DialogContent>
      </Dialog>
    </>
  );
}
