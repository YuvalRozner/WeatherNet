import * as React from "react";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import StepContent from "@mui/material/StepContent";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import { getContent } from "../../../utils/staticData/aboutContent";
import {
  AboutContainer,
  AboutContentBox,
  NextLabelButtonContainer,
  LabelNavigateButton,
  ClickableImage,
} from "./about.style";
import { useTheme, Tooltip, Skeleton } from "@mui/material";
import ImageDialog from "../../dataDisplays/imageDialog";

export default function VerticalLinearStepper() {
  const [activeStep, setActiveStep] = React.useState(0);
  const [open, setOpen] = React.useState(false);
  const [selectedImage, setSelectedImage] = React.useState(null);
  const [imageLoaded, setImageLoaded] = React.useState(false);
  const theme = useTheme();
  const themeMode = theme.palette.mode;
  const content = getContent(themeMode);

  React.useEffect(() => {
    setImageLoaded(false);
  }, [activeStep]);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
  };

  const handleImageClick = (imageData) => {
    if (imageData.image && imageData.imageTitle && imageData.imageDescription) {
      setSelectedImage(imageData);
      setOpen(true);
    }
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
  };

  return (
    <>
      <AboutContainer>
        <AboutContentBox>
          <Stepper activeStep={activeStep} orientation="vertical">
            {content.map((step, index) => (
              <Step key={step.label}>
                <StepLabel>
                  <Typography variant="h6">
                    <b>{step.label}</b>
                  </Typography>
                </StepLabel>
                <StepContent>
                  <Typography>
                    {step.description.split("\n").map((line, idx) => (
                      <React.Fragment key={idx}>
                        {line.split("\b").map((part, subIdx) => (
                          <React.Fragment key={subIdx}>
                            {subIdx % 2 === 1 ? <b>{part}</b> : part}
                          </React.Fragment>
                        ))}
                        <br />
                      </React.Fragment>
                    ))}
                  </Typography>
                  <NextLabelButtonContainer>
                    <LabelNavigateButton
                      variant="contained"
                      onClick={handleNext}
                    >
                      {index === content.length - 1 ? "Finish" : "Continue"}
                    </LabelNavigateButton>
                    <LabelNavigateButton
                      disabled={index === 0}
                      onClick={handleBack}
                    >
                      Back
                    </LabelNavigateButton>
                  </NextLabelButtonContainer>
                </StepContent>
              </Step>
            ))}
          </Stepper>
          {activeStep === content.length && (
            <Paper square elevation={0} style={{ padding: 3 }}>
              <Typography>That was WeatherNet!</Typography>
              <LabelNavigateButton onClick={handleReset}>
                Explore WeatherNet again
              </LabelNavigateButton>
            </Paper>
          )}
        </AboutContentBox>
        {activeStep !== content.length &&
          (content[activeStep].image &&
          content[activeStep].imageTitle &&
          content[activeStep].imageDescription ? (
            <>
              {!imageLoaded && (
                <Skeleton
                  variant="rectangular"
                  width="100%"
                  height={200}
                  style={{ borderRadius: "4px", marginBottom: "16px" }}
                />
              )}
              <Tooltip title="Click to open wider, with description">
                <ClickableImage
                  src={content[activeStep].image}
                  alt={content[activeStep].label}
                  onClick={() => handleImageClick(content[activeStep])}
                  onLoad={handleImageLoad}
                  style={{
                    cursor: "pointer",
                    display: imageLoaded ? "block" : "none",
                  }}
                />
              </Tooltip>
            </>
          ) : (
            <>
              {!imageLoaded &&
                content[activeStep].image &&
                content[activeStep].image !== " " && (
                  <Skeleton
                    variant="rectangular"
                    width="100%"
                    height={200}
                    style={{ borderRadius: "4px", marginBottom: "16px" }}
                  />
                )}
              {content[activeStep].image &&
              content[activeStep].image !== " " ? (
                <ClickableImage
                  src={content[activeStep].image}
                  alt={content[activeStep].label}
                  onLoad={handleImageLoad}
                  style={{
                    cursor: "default",
                    display: imageLoaded ? "block" : "none",
                  }}
                />
              ) : null}
            </>
          ))}
      </AboutContainer>
      <ImageDialog
        open={open}
        handleClose={() => setOpen(false)}
        image={selectedImage ? selectedImage.image : ""}
        title={selectedImage ? selectedImage.imageTitle : ""}
        description={selectedImage ? selectedImage.imageDescription : ""}
        isDescriptionAbove={true}
      />
    </>
  );
}
