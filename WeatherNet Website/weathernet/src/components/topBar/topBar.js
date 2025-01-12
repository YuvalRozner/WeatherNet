import React from 'react';
import { TopBarContainer, Logo, SiteName } from './topBar.style';

const TopBar = () => {
  return (
    <TopBarContainer>
      <Logo src="/logo/logo_empty.png" alt="WeatherNet Logo" />
      <SiteName>WeatherNet</SiteName>
    </TopBarContainer>
  );
};

export default TopBar;
