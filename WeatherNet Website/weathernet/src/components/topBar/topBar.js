import React from 'react';
import { TopBarContainer, Logo, SiteName, ShareButtonsContainer } from './topBar.style';
import ThemeToggle from '../themeToggle/themeToggle';
import ShareContainer from '../shareContainer/shareContainer';

const TopBar = ({ toggleTheme, theme }) => {
  const shareUrl = window.location.href; // URL to share
  const title = 'Check out WeatherNet!';

  return (
    <TopBarContainer>
      <Logo src="/logo/logo_empty.png" alt="WeatherNet Logo" />
      <SiteName>WeatherNet</SiteName>
      <ShareButtonsContainer>
        <ShareContainer shareUrl={shareUrl} title={title} theme={theme} />
      </ShareButtonsContainer>
      <ThemeToggle toggleTheme={toggleTheme} theme={theme} />
    </TopBarContainer>
  );
};

export default TopBar;
