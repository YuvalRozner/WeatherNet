import React from "react";
import { StyledGitHubIconButton, StyledGitHubIcon } from "./topBar.style.js";
import Tooltip from "@mui/material/Tooltip";

export default function GitHubButton() {
  const handleGitHubClick = () => {
    window.open("https://github.com/YuvalRozner/WeatherNet", "_blank");
  };

  return (
    <Tooltip title="Go to WeatherNet Repo" arrow>
      <StyledGitHubIconButton onClick={handleGitHubClick}>
        <StyledGitHubIcon />
      </StyledGitHubIconButton>
    </Tooltip>
  );
}
