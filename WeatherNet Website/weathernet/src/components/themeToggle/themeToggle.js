import React from 'react';
import { ToggleContainer, LightIcon, DarkIcon } from './themeToggle.style';
import { FaSun, FaMoon } from 'react-icons/fa';

const ThemeToggle = ({ toggleTheme, theme }) => {
  return (
    <ToggleContainer onClick={toggleTheme}>
      <DarkIcon $isVisible={theme === 'dark'}>
        <FaMoon />
      </DarkIcon>
      <LightIcon $isVisible={theme === 'light'}>
        <FaSun />
      </LightIcon>
    </ToggleContainer>
  );
};

export default ThemeToggle;
