import React from "react";
import { StyledIconButton, StyledGitHubIcon } from "./topBar.style.js";

export default function GitHubButton() {
  const handleGitHubClick = () => {
    window.open("https://github.com/YuvalRozner/WeatherNet", "_blank");
  };

  return (
    <StyledIconButton onClick={handleGitHubClick}>
      <StyledGitHubIcon />
    </StyledIconButton>
  );
}
