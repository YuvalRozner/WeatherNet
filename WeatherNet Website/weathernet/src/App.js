import React, { useState, useEffect } from 'react';
import './App.css';
import TopBar from './components/topBar/topBar';
import { ThemeProvider } from 'styled-components';
import { lightTheme, darkTheme } from './utils/theme';
import Body from './components/body/body';

function App() {
  const [theme, setTheme] = useState(() => {
    // Check localStorage for a saved theme
    const savedTheme = localStorage.getItem('theme');
    return savedTheme ? savedTheme : 'dark'; // Default to 'dark' if no saved theme
  });

  useEffect(() => {
    // Save the theme to localStorage whenever it changes
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  return (
    <ThemeProvider theme={theme === 'light' ? lightTheme : darkTheme}>
      <TopBar toggleTheme={toggleTheme} theme={theme} />
      <Body />
    </ThemeProvider>
  );
}

export default App;
