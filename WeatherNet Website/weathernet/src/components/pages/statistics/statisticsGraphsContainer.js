import { useState, useEffect } from "react";
import { useTheme } from "@mui/material/styles";
import Divider from "@mui/material/Divider";
import Skeleton from "@mui/material/Skeleton";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import {
  getStatisticsData,
  getStatisticsModelsGraphsData,
} from "../../../utils/staticData/statisticsData";
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
  const statisticsModelsGraphsData = getStatisticsModelsGraphsData(themeMode);

  useEffect(() => {
    setImagesLoaded(
      new Array(statisticsData.length + statisticsModelsGraphsData.length).fill(
        false
      )
    );
  }, [statisticsData.length, statisticsModelsGraphsData.length]);

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
            <Divider
              textAlign="left"
              variant="middle"
              sx={{ fontSize: "1.3rem" }}
            >
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

      <Divider
        textAlign="left"
        variant="middle"
        sx={{ fontSize: "1.3rem", marginTop: "8px", marginBottom: "18px" }}
      >
        {imagesLoaded
          .slice(statisticsData.length, statisticsData.length + 4)
          .every(Boolean) ? (
          <b>Models Time Series Graphs</b>
        ) : (
          <Skeleton width="40%" sx={{ borderRadius: "4px" }} />
        )}
      </Divider>
      <Typography variant="body1" sx={{ marginBottom: "18px" }}>
        The graphs below present a segment of the validation set, where each
        model independently predicts temperature within its designated time
        period. <br />
        Each graph illustrates the predicted and actual temperatures over a
        100-hour span. <br />
        As expected, the model trained for the next 12 hours delivers the most
        accurate predictions, while accuracy gradually decreases for models
        forecasting further into the future.
      </Typography>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "space-around",
        }}
      >
        {statisticsModelsGraphsData.map((data, index) => (
          <div key={index} style={{ width: "45%", marginBottom: "20px" }}>
            {index < statisticsModelsGraphsData.length && (
              <Divider
                textAlign="left"
                variant="middle"
                sx={{ fontSize: "1.1rem" }}
              >
                {imagesLoaded[index + statisticsData.length] ? (
                  <b>{data.title}</b>
                ) : (
                  <Skeleton width="40%" sx={{ borderRadius: "4px" }} />
                )}
              </Divider>
            )}
            <ImageContainer>
              {!imagesLoaded[index + statisticsData.length] ? (
                <Skeleton
                  variant="rectangular"
                  width="100%"
                  height={118}
                  sx={{ borderRadius: "4px" }}
                />
              ) : (
                <Tooltip title="Click to open wider">
                  <img
                    src={data.image}
                    alt={data.title}
                    style={{
                      width: "100%",
                      cursor: "pointer",
                    }}
                    onClick={() => handleClickOpen(data)}
                  />
                </Tooltip>
              )}
              <HiddenImage
                src={data.image}
                alt={data.title}
                onLoad={() => handleImageLoad(index + statisticsData.length)}
              />
            </ImageContainer>
          </div>
        ))}
      </div>

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
