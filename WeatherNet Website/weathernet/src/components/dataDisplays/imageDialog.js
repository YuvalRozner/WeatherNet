import React, { useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Button from "@mui/material/Button";
import Skeleton from "@mui/material/Skeleton";
import styled from "styled-components";

const DialogImage = styled.img`
  max-height: 700px;
  max-width: 100%;
  height: auto; // Ensures the image maintains its original aspect ratio
  display: ${(props) => (props.loaded ? "block" : "none")};
  margin: 0 auto; // Centers the image horizontally
`;

export default function ImageDialog({
  open,
  handleClose,
  image,
  title,
  description,
  isDescriptionAbove = false,
}) {
  const [dialogImageLoaded, setDialogImageLoaded] = useState(false);

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        {title ? (
          <b>{title}</b>
        ) : (
          <Skeleton width="60%" sx={{ borderRadius: "4px" }} />
        )}
        <Button
          onClick={handleClose}
          sx={{
            position: "absolute",
            right: "10px",
            top: "10px",
            minWidth: "2.8rem",
            borderRadius: "50%",
          }}
        >
          <span style={{ fontSize: "2.2rem", lineHeight: "1" }}>×</span>
        </Button>
      </DialogTitle>
      <DialogContent>
        {!dialogImageLoaded && (
          <Skeleton
            variant="rectangular"
            width="100%"
            height={200}
            sx={{ borderRadius: "4px" }}
          />
        )}
        {dialogImageLoaded && description && isDescriptionAbove && (
          <DialogContentText style={{ fontSize: "1.2rem" }}>
            {description.split("\n").map((line, idx) => (
              <React.Fragment key={idx}>
                {line}
                <br />
              </React.Fragment>
            ))}
          </DialogContentText>
        )}
        {image && (
          <DialogImage
            src={image}
            alt={title}
            loaded={dialogImageLoaded}
            onLoad={() => setDialogImageLoaded(true)}
          />
        )}
        {dialogImageLoaded && description && !isDescriptionAbove && (
          <DialogContentText style={{ fontSize: "1.2rem" }}>
            {description.split("\n").map((line, idx) => (
              <React.Fragment key={idx}>
                {line}
                <br />
              </React.Fragment>
            ))}
          </DialogContentText>
        )}
      </DialogContent>
    </Dialog>
  );
}
