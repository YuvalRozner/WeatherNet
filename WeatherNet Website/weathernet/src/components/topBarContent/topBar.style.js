import styled from "styled-components";
import { FaGithub as GitHubIcon } from "react-icons/fa";
import IconButton from "@mui/material/IconButton";

export const SiteNameContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const SiteName = styled.h1`
  color: ${(props) => props.theme.text};
  font-size: 1.7rem;
  margin: 0;

  &:hover {
    color: ${(props) => props.theme.hoverText};
  }
`;

export const Logo = styled.img`
  height: 60px;
  margin: 0px 22px 0px 6px;
  padding: 2px 0px;
`;

export const InvisibleDiv = styled.div`
  width: 54px;
`;

export const ShareButtonsContainer = styled.div`
  display: flex;
  align-items: center;
  margin-left: auto;
`;

export const StyledGitHubIcon = styled(GitHubIcon)`
  color: #58a6ff;
  font-size: 32px;
`;

export const StyledIconButton = styled(IconButton)`
  width: 45px;
  height: 45px;
  padding: 7px;
`;
