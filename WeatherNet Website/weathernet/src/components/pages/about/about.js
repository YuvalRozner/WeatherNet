import * as React from "react";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import StepContent from "@mui/material/StepContent";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import { content } from "../../../utils/aboutContent";
import {
  AboutContainer,
  AboutContentBox,
  NextLabelButtonContainer,
  LabelNavigateButton,
  LabelImageContainer,
} from "./about.style";

export default function VerticalLinearStepper() {
  const [activeStep, setActiveStep] = React.useState(0);

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
    <AboutContainer>
      <AboutContentBox>
        <Stepper activeStep={activeStep} orientation="vertical">
          {content.map((step, index) => (
            <Step key={step.label}>
              <StepLabel // TODO: decide if we want to show this
                optional={
                  index === content.length - 1 ? (
                    <Typography variant="caption">Last step</Typography>
                  ) : null
                }
              >
                {step.label}
              </StepLabel>
              <StepContent>
                <Typography>{step.description}</Typography>
                <NextLabelButtonContainer>
                  <LabelNavigateButton variant="contained" onClick={handleNext}>
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
            <Typography> {/* TODO: change? */} Thats all for now!</Typography>
            <LabelNavigateButton onClick={handleReset}>
              Reset
            </LabelNavigateButton>
          </Paper>
        )}
      </AboutContentBox>
      {activeStep !== content.length && (
        <LabelImageContainer src={content[activeStep].image} alt="about" />
      )}
    </AboutContainer>
  );
}
