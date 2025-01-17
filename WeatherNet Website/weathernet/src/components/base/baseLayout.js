import { LayoutContainer, AppLogo } from "./baseLayout.style.js";

export function BaseLayout() {
  return (
    <LayoutContainer>
      <AppLogo src="/logo/logo_empty.png" alt="logo" />
    </LayoutContainer>
  );
}
