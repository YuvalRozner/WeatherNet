import {
  Logo,
  SiteName,
  SiteNameContainer,
  ShareButtonsContainer,
  InvisibleDiv,
  IconAndTitleContainer,
} from "./topBar.style.js";
import ShareContainer from "./shareContainer.js";
import { ThemeSwitcher } from "@toolpad/core";
import GitHubButton from "./githubContainer.js";
import { useNavigate } from "react-router-dom";

export function AppTitle() {
  const navigate = useNavigate();
  const handleIconClick = () => {
    navigate(`/Home`); // Navigate to the segment
  };

  return (
    <IconAndTitleContainer onClick={handleIconClick}>
      <Logo src="/logo/compressed_empty_logo.png" alt="WeatherNet Logo" />
      <SiteNameContainer>
        <SiteName>WeatherNet</SiteName>
      </SiteNameContainer>
    </IconAndTitleContainer>
  );
}

export function ToolbarActions() {
  const shareUrl = window.location.href; // URL to share
  const title = "Check out WeatherNet!";
  return (
    <>
      <ThemeSwitcher sx={{ margin: "0px", padding: "0px" }} />
      <GitHubButton style={{ margin: "0px", padding: "0px" }} />
      <InvisibleDiv />
      <ShareButtonsContainer>
        <ShareContainer shareUrl={shareUrl} title={title} />
      </ShareButtonsContainer>
    </>
  );
}
