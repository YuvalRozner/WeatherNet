import React from 'react';
import { App, AppHeader, AppLogo } from './body.style.js'; // Import the styled components

function Body() {
  return (
    <App>
      <AppHeader>
        <AppLogo src="/logo/compressed_logo.png" alt="logo" />
        <h1>This Is WeatherNet!</h1>
        <p>Your reliable weather forecast source</p>
      </AppHeader>
    </App>
  );
}

export default Body; 