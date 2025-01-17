import { LayoutContainer, AppLogo } from "./baseLayout.style.js";

export function BaseLayout() {
  return (
    <LayoutContainer>
      <AppLogo src="/logo/compressed_empty_logo.png" alt="logo" />
    </LayoutContainer>
  );
}
