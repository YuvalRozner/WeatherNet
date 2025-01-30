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
  LabelImageContainer,
} from "./about.style";
import { useTheme } from "@mui/material";

export default function VerticalLinearStepper() {
  const [activeStep, setActiveStep] = React.useState(0);
  const theme = useTheme();
  const themeMode = theme.palette.mode;
  const content = getContent(themeMode);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
  };

  return (
    <>
      <AboutContainer>
        <AboutContentBox>
          <Stepper activeStep={activeStep} orientation="vertical">
            {content.map((step, index) => (
              <Step key={step.label}>
                <StepLabel
                // optional={
                //   index === content.length - 1 ? (
                //     <Typography variant="caption">Last step</Typography>
                //   ) : null
                // }
                >
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
            <Paper square elevation={0} style={{ p: 3 }}>
              <Typography> That was WeatherNet!</Typography>
              <LabelNavigateButton onClick={handleReset}>
                Explore WeatherNet again
              </LabelNavigateButton>
            </Paper>
          )}
        </AboutContentBox>
        {activeStep !== content.length && (
          <LabelImageContainer
            src={content[activeStep].image}
            alt={content[activeStep].image}
          />
        )}
      </AboutContainer>
    </>
  );
}
