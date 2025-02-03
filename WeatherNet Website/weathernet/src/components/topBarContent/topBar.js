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
import { useState } from "react";
import Skeleton from "@mui/material/Skeleton";
import Settings from "./settings.js";

export function AppTitle() {
  const navigate = useNavigate();
  const [logoLoaded, setLogoLoaded] = useState(false);

  const handleIconClick = () => {
    navigate(`/Home`); // Navigate to the segment
  };

  const handleLogoLoad = () => {
    setLogoLoaded(true);
  };

  return (
    <IconAndTitleContainer onClick={handleIconClick}>
      {!logoLoaded && (
        <Skeleton
          variant="circular"
          width={56}
          height={56}
          style={{ margin: "0px 22px 0px 6px" }}
        />
      )}
      <Logo
        src="/logo/compressed_empty_logo.png"
        alt="WeatherNet Logo"
        onLoad={handleLogoLoad}
        style={{ display: logoLoaded ? "block" : "none" }}
      />
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
      <Settings />
      <ThemeSwitcher sx={{ margin: "0px", padding: "0px" }} />
      <GitHubButton style={{ margin: "0px", padding: "0px" }} />
      <InvisibleDiv />
      <ShareButtonsContainer>
        <ShareContainer shareUrl={shareUrl} title={title} />
      </ShareButtonsContainer>
    </>
  );
}
