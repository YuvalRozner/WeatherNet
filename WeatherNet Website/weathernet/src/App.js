import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={process.env.PUBLIC_URL + '/logo/logo.png'} className="App-logo" alt="logo" />
        <h1>This Is WeatherNet!!!</h1>
        <p>Your reliable weather forecast source</p>
      </header>
    </div>
  );
}

export default App;