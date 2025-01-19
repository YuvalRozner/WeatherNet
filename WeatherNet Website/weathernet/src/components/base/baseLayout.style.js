import styled, { keyframes } from "styled-components";
import { styled as muiStyled } from "@mui/material/styles";
import ContactSupportOutlined from "@mui/icons-material/ContactSupportOutlined";

const AppLogoSpin = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

export const AppLogo = styled.img`
  height: 60vmin;
  pointer-events: none;

  @media (prefers-reduced-motion: no-preference) {
    animation: ${AppLogoSpin} infinite 15s linear;
  }
`;

export const LayoutContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

export const HelpButtonContainer = styled.button`
  margin-top: auto;
  margin-left: auto;
  background-color: transparent;
  border: none;
  cursor: pointer;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
`;

export const HelpIcon = muiStyled(ContactSupportOutlined)`
  font-size: 40px !important;
  color: ${({ theme }) => theme.palette.action.disabled};
  &:hover {
    color: ${({ theme }) => theme.palette.action.helpButtonHover};
  }
`;
