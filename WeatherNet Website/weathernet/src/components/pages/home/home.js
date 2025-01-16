import { Appi, AppHeader, AppLogo } from "./home.style.js";

export function Home() {
  return (
    <Appi>
      <AppHeader>
        <AppLogo src="/logo/compressed_logo.png" alt="logo" />
        <h1>This Is WeatherNet!</h1>
        <p>Your reliable weather forecast source</p>
      </AppHeader>
    </Appi>
  );
}
