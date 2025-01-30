import { useState, useEffect } from "react";
import { useTheme } from "@mui/material/styles";
import Divider from "@mui/material/Divider";
import Skeleton from "@mui/material/Skeleton";
import Tooltip from "@mui/material/Tooltip";
import { getStatisticsData } from "../../../utils/staticData/statisticsData";
import {
  Container,
  ImageContainer,
  Description,
  HiddenImage,
} from "./statistics.style";
import React from "react";
import ImageDialog from "../../dataDisplays/imageDialog";

export default function StatisticsGraphsContainer() {
  const [open, setOpen] = useState(false);
  const [selectedData, setSelectedData] = useState(null);
  const [imagesLoaded, setImagesLoaded] = useState([]);
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
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setSelectedData(null);
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
              <Tooltip title="Click to open wider">
                <img
                  src={data.image}
                  alt={data.title}
                  style={{
                    marginRight: "20px",
                    width: "65%",
                    cursor: "pointer",
                  }}
                  onClick={() => handleClickOpen(data)}
                />
              </Tooltip>
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

      <ImageDialog
        open={open}
        handleClose={handleClose}
        image={selectedData ? selectedData.image : ""}
        title={selectedData ? selectedData.title : ""}
        description={selectedData ? selectedData.description : ""}
      />
    </Container>
  );
}
