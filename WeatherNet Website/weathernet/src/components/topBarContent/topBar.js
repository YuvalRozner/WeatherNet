import {
  Logo,
  SiteName,
  SiteNameContainer,
  ShareButtonsContainer,
  InvisibleDiv,
} from "./topBar.style.js";
import ShareContainer from "./shareContainer.js";
import { ThemeSwitcher } from "@toolpad/core";

export function AppTitle() {
  return (
    <>
      <Logo src="/logo/logo_empty.png" alt="WeatherNet Logo" />
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
      <InvisibleDiv />
      <ShareButtonsContainer>
        <ShareContainer shareUrl={shareUrl} title={title} />
      </ShareButtonsContainer>
    </>
  );
}
