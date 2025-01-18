import { useState } from "react";
import { LayoutContainer, AppLogo } from "./baseLayout.style.js";
import { SkeletonLayout } from "./skeltonLayot.js";

export function BaseLayout() {
  const [imgLoaded, setImgLoaded] = useState(false);

  return (
    <LayoutContainer>
      {!imgLoaded && <SkeletonLayout />}
      <AppLogo
        src="/logo/compressed_empty_logo.png"
        alt="logo"
        onLoad={() => setImgLoaded(true)}
        style={{ display: imgLoaded ? "block" : "none" }}
      />
    </LayoutContainer>
  );
}
