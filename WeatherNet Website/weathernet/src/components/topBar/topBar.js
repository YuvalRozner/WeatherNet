import React from 'react';
import { TopBarContainer, Logo, SiteName, ThemeToggleButton } from './topBar.style';
import { FaSun, FaMoon } from 'react-icons/fa';

const TopBar = ({ toggleTheme, theme }) => {
  return (
    <TopBarContainer>
      <Logo src="/logo/logo_empty.png" alt="WeatherNet Logo" />
      <SiteName>WeatherNet</SiteName>
      <ThemeToggleButton onClick={toggleTheme}>
        {theme === 'light' ? <FaMoon /> : <FaSun />}
      </ThemeToggleButton>
    </TopBarContainer>
  );
};

export default TopBar;
