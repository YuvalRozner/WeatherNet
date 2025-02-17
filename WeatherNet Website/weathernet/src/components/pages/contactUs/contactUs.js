import * as React from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import { DraggablePaperComponent } from "../../../utils/logicForComponents/draggable";
import logger from "../../../utils/logicForComponents/logger";
import {
  DraggableTopArea,
  DraggableDialogActions,
  DraggableDialogTitle,
} from "./contactUs.style";
import { WeatherNetLogoLayout } from "../../base/weathenetLogoLayout";
import { useNotifications } from "@toolpad/core/useNotifications";

export default function FormDialog() {
  const [open, setOpen] = React.useState(true);
  const notifications = useNotifications();

  const handleClose = () => {
    setOpen(false);
    window.history.back();
  };

  return (
    <React.Fragment>
      <WeatherNetLogoLayout />
      <Dialog
        open={open}
        onClose={handleClose}
        PaperComponent={DraggablePaperComponent}
        PaperProps={{
          component: "form",
          onSubmit: (event) => {
            event.preventDefault();
            const formData = new FormData(event.currentTarget);
            const formJson = Object.fromEntries(formData.entries());
            logger.log(`got message from user: ${JSON.stringify(formJson)}`);
            handleClose();
            notifications.show(
              "Message sent. We will get back to you as soon as possible.",
              {
                autoHideDuration: 3000,
              }
            );
          },
        }}
      >
        <DraggableTopArea>
          <DraggableDialogTitle id="draggable-dialog-title">
            Send Message
          </DraggableDialogTitle>
        </DraggableTopArea>
        <DialogContent>
          <DialogContentText>
            Please enter your email address, subject, and message below.
          </DialogContentText>
          <DialogContentText>
            We will get back to you as soon as possible.
          </DialogContentText>
          <TextField
            required
            margin="dense"
            id="email"
            name="email"
            label="Your Email Address"
            type="email"
            fullWidth
            variant="standard"
          />
          <TextField
            required
            margin="dense"
            id="subject"
            name="subject"
            label="Subject"
            type="text"
            fullWidth
            variant="standard"
          />
          <TextField
            required
            margin="dense"
            id="message"
            name="message"
            label="Message"
            type="text"
            fullWidth
            variant="standard"
            multiline
            rows={3}
          />
        </DialogContent>
        <DraggableDialogActions id="draggable-dialog-title">
          <Button onClick={handleClose}>Cancel</Button>
          <Button type="submit">Send</Button>
        </DraggableDialogActions>
      </Dialog>
    </React.Fragment>
  );
}
