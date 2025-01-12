import React, { useState, useEffect } from 'react';
import './App.css';
import TopBar from './components/topBar/topBar';
import { ThemeProvider } from 'styled-components';
import { lightTheme, darkTheme } from './utils/theme';

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
      <div className="App">
        <header className="App-header">
          <img src="/logo/compressed_logo.png" className="App-logo" alt="logo" />
          <h1>This Is WeatherNet!</h1>
          <p>Your reliable weather forecast source</p>
        </header>
      </div>
    </ThemeProvider>
  );
}

export default App;
