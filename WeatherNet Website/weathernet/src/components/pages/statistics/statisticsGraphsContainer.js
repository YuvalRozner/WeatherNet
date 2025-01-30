import { useState, useEffect } from "react";
import { useTheme } from "@mui/material/styles";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import Skeleton from "@mui/material/Skeleton";
import { getStatisticsData } from "../../../utils/staticData/statisticsData";
import styled from "styled-components";
import {
  Container,
  ImageContainer,
  Description,
  HiddenImage,
} from "./statistics.style";
import React from "react";

const DialogImage = styled.img`
  width: 100%;
  display: ${(props) => (props.loaded ? "block" : "none")};
`;

export default function StatisticsGraphsContainer() {
  const [open, setOpen] = useState(false);
  const [selectedData, setSelectedData] = useState(null);
  const [imagesLoaded, setImagesLoaded] = useState([]);
  const [dialogImageLoaded, setDialogImageLoaded] = useState(false);
  const theme = useTheme();
  const themeMode = theme.palette.mode;

  const statisticsData = getStatisticsData(themeMode);

  useEffect(() => {
    setImagesLoaded(new Array(statisticsData.length).fill(false));
  }, [statisticsData.length]);

  const handleImageLoad = (index) => {
    setImagesLoaded((prev) => {
      const updated = [...prev];
      updated[index] = true;
      return updated;
    });
  };

  const handleClickOpen = (data) => {
    setSelectedData(data);
    setDialogImageLoaded(false);
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setSelectedData(null);
    setDialogImageLoaded(false);
  };

  return (
    <Container>
      {statisticsData.map((data, index) => (
        <div key={index}>
          {index < statisticsData.length && (
            <Divider textAlign="left" variant="middle">
              {imagesLoaded[index] ? (
                <b>{data.title}</b>
              ) : (
                <Skeleton width="40%" sx={{ borderRadius: "4px" }} />
              )}
            </Divider>
          )}
          <ImageContainer>
            {!imagesLoaded[index] ? (
              <Skeleton
                variant="rectangular"
                width="65%"
                height={118}
                style={{ marginRight: "20px" }}
                sx={{ borderRadius: "4px" }}
              />
            ) : (
              <img
                src={data.image}
                alt={data.title}
                style={{ marginRight: "20px", width: "65%", cursor: "pointer" }}
                onClick={() => handleClickOpen(data)}
              />
            )}
            {imagesLoaded[index] ? (
              <Description>
                {data.description.split("\n").map((line, idx) => (
                  <React.Fragment key={idx}>
                    {line}
                    <br />
                  </React.Fragment>
                ))}
              </Description>
            ) : (
              <div style={{ flex: 1 }}>
                <Skeleton width="80%" sx={{ borderRadius: "4px" }} />
                <Skeleton width="60%" sx={{ borderRadius: "4px" }} />
                <Skeleton width="40%" sx={{ borderRadius: "4px" }} />
              </div>
            )}
            <HiddenImage
              src={data.image}
              alt={data.title}
              onLoad={() => handleImageLoad(index)}
            />
          </ImageContainer>
        </div>
      ))}

      <Dialog open={open} onClose={handleClose} maxWidth="lg" fullWidth>
        <DialogTitle>
          {selectedData ? (
            <b>{selectedData.title}</b>
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
          {selectedData && (
            <DialogImage
              src={selectedData.image}
              alt={selectedData.title}
              loaded={dialogImageLoaded}
              onLoad={() => setDialogImageLoaded(true)}
            />
          )}
          {dialogImageLoaded && selectedData && (
            <DialogContentText style={{ fontSize: "1.2rem" }}>
              {selectedData.description.split("\n").map((line, idx) => (
                <React.Fragment key={idx}>
                  {line}
                  <br />
                </React.Fragment>
              ))}
            </DialogContentText>
          )}
        </DialogContent>
      </Dialog>
    </Container>
  );
}
