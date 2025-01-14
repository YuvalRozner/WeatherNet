import React from 'react';
import { ToggleContainer, ThemeIcon } from './themeToggle.style';
import { FaSun, FaMoon } from 'react-icons/fa';

const ThemeToggle = ({ toggleTheme, theme }) => {
  return (
    <ToggleContainer onClick={toggleTheme}>
      <ThemeIcon $isVisible={theme === 'dark'}>
        {theme === 'dark' && <FaMoon />}
        {theme === 'light' && <FaSun />}
      </ThemeIcon>
    </ToggleContainer>
  );
};

export default ThemeToggle;
