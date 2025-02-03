import styled from "styled-components";
import { FaGithub as GitHubIcon } from "react-icons/fa";
import IconButton from "@mui/material/IconButton";
import Box from "@mui/material/Box";
import { MyTheme } from "../../utils/theme";
import ShareIcon from "@mui/icons-material/Share";

export const SiteNameContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const SiteName = styled.h1`
  font-size: 1.7rem;
  margin: 0;

  @media (max-width: ${MyTheme.breakpoints.values.sm}px) {
    font-size: 1.2rem;
    margin: 0;
  }

  &:hover {
    transform: scale(1.03);
    transition: transform 0.2s ease-in-out;
  }
`;

export const Logo = styled.img`
  height: 60px;
  margin: 0px 22px 0px 6px;
  padding: 2px 0px;

  @media (max-width: ${MyTheme.breakpoints.values.sm}px) {
    height: 36px;
    margin: 0px 4px 0px 0px;
    padding: 0px 0px;
  }

  &:hover {
    transform: scale(1.03);
    transition: transform 0.2s ease-in-out;
  }
`;

export const InvisibleDiv = styled.div`
  width: 54px;

  @media (max-width: ${MyTheme.breakpoints.values.sm}px) {
    width: 15px;
  }
`;

export const ShareButtonsContainer = styled.div`
  display: flex;
  align-items: center;
`;

export const StyledGitHubIcon = styled(GitHubIcon)`
  color: #58a6ff;
  font-size: 32px;
  padding: 0px;
  margin: 0px;

  @media (max-width: ${MyTheme.breakpoints.values.sm}px) {
    font-size: 28px;
  }
`;

export const StyledGitHubIconButton = styled(IconButton)`
  width: 45px;
  height: 45px;
  padding: 0px !important;
  margin: 0 0 0 8px !important;
`;

export const IconAndTitleContainer = styled.div`
  display: flex;
  align-items: center;
  cursor: pointer;
`;

export const ShareContainerBox = styled(Box)`
  position: absolute;
  top: 11px;
  right: 30px;

  @media (max-width: ${MyTheme.breakpoints.values.sm}px) {
    top: 12px;
    right: 9px;
  }
`;

export const StyledShareIcon = styled(ShareIcon)`
  width: 24px;
  height: 24px;

  @media (max-width: ${MyTheme.breakpoints.values.sm}px) {
    width: 20px !important;
    height: 20px !important;
  }
`;
