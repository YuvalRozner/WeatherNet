import styled from "styled-components";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";

export const AboutContainer = styled(Box)`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2;
`;

export const AboutContentBox = styled(Box)`
  width: 730px;
`;

export const NextLabelButtonContainer = styled(Box)`
  margin-bottom: 2;
`;

export const LabelNavigateButton = styled(Button)`
  margin-top: 1;
  margin-right: 1;
`;

export const LabelImageContainer = styled.img`
  width: 100%;
`;
