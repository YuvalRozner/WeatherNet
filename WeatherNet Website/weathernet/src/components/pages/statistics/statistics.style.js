import styled from "styled-components";
import Button from "@mui/material/Button";

export const Container = styled.div`
  /* Add any container-specific styles if needed */
`;

export const ImageContainer = styled.div`
  display: flex;
  margin-bottom: 16px;
  margin-top: 16px;
  align-items: center;
`;

export const Description = styled.p``;

export const HiddenImage = styled.img`
  display: none;
`;

export const StyledButton = styled(Button)`
  /* Add any additional styles for buttons if needed */
`;

export const CloseButton = styled(Button)`
  position: absolute;
  right: 10px;
  top: 10px;
  min-width: 2.1rem !important;
  padding: 0 !important;
  border-radius: 50% !important;

  span {
    font-size: 2.1rem;
    line-height: 1;
  }
`;
