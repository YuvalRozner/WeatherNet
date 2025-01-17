import {
  Logo,
  SiteName,
  SiteNameContainer,
  ShareButtonsContainer,
  InvisibleDiv,
} from "./topBar.style.js";
import ShareContainer from "./shareContainer.js";
import { ThemeSwitcher } from "@toolpad/core";
import GitHubButton from "./githubContainer.js";

export function AppTitle() {
  return (
    <>
      <Logo src="/logo/compressed_empty_logo.png" alt="WeatherNet Logo" />
      <SiteNameContainer>
        <SiteName>WeatherNet</SiteName>
      </SiteNameContainer>
    </>
  );
}

export function ToolbarActions() {
  const shareUrl = window.location.href; // URL to share
  const title = "Check out WeatherNet!";
  return (
    <>
      <ThemeSwitcher />
      <GitHubButton />
      <InvisibleDiv />
      <ShareButtonsContainer>
        <ShareContainer shareUrl={shareUrl} title={title} />
      </ShareButtonsContainer>
    </>
  );
}
