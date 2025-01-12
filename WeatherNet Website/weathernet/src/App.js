import React from 'react';
import './App.css';
import TopBar from './components/topBar/topBar';

function App() {
  return (
    <div className="App">
      <TopBar />
      <header className="App-header">
        <img src="/logo/compressed_logo.png" className="App-logo" alt="logo" />
        <h1>This Is WeatherNet!</h1>
        <p>Your reliable weather forecast source</p>
      </header>
    </div>
  );
}

export default App;
