# WeatherNet

<div style="display: flex; align-items: center;">
    <!-- Image Container -->
    <div style="flex: 1; text-align: center;">
        <img src="/logo/pic4.png" alt="Weather Prediction Icon" width="250">
    </div>

    <div style="flex: 2;">
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th style="padding: 8px; text-align: left;">Module</th>
                <th style="padding: 8px; text-align: left;">Link</th>
            </tr>
            <tr>
                <td style="padding: 8px;">Website</td>
                <td style="padding: 8px;"><a href="https://weathernet-il.web.app">weathernet-il.web.app</a></td>
            </tr>
            <tr>
                <td style="padding: 8px;">Github</td>
                <td style="padding: 8px;"><a href="https://github.com/YuvalRozner/WeatherNet">WeatherNet Project</a></td>
            </tr>
            <tr>
                <td style="padding: 8px;">User Manual</td>
                <td style="padding: 8px;"><a href="#">User Manual</a></td>
            </tr>
            <tr>
                <td style="padding: 8px;">Developer Manual</td>
                <td style="padding: 8px;"><a href="#">Developer Manual</a></td>
            </tr>
            <tr>
                <td style="padding: 8px;">Phase A Paper</td>
                <td style="padding: 8px;"><a href="#">Link</a></td>
            </tr>
            <tr>
                <td style="padding: 8px;">Phase B Paper</td>
                <td style="padding: 8px;"><a href="#">Link</a></td>
            </tr>
        </table>
    </div>

</div>

## About WeatherNet

WeatherNet is an advanced weather forecasting system designed to provide **accurate mid-term temperature predictions for Israel**. It leverages **machine learning** to analyze historical and real-time data, delivering reliable forecasts through a user-friendly web interface.

## Project Phases

The development of WeatherNet was divided into two primary phases:

- **Phase A (Research & Proof of Concept - POC)**:  
  Conducted in-depth research on weather forecasting methods and machine learning techniques, culminating in a proof-of-concept model to validate feasibility.

- **Phase B (Development & Implementation)**:  
  Transitioned from research to full-scale implementation, including **training the ML model, developing the user interface, and deploying the system**.

## System Components

WeatherNet consists of two primary components:

- **Backend Machine Learning Model**:  
  Processes weather data, trains on historical and real-time information, and generates accurate predictions.

- **Frontend Web-Based Platform**:  
  Provides an interactive interface for users to access forecasts, compare results, and explore model insights.

## Machine Learning Architecture

<div style="text-align: center;">
    <img src="/logo/architecture_dark_framed.png" alt="ML Architecture" style="width: 80%;">
</div>

Our model follows a **hybrid approach** combining:

- **1D Convolutional Neural Networks (CNNs)** for feature extraction.
- **Positional Encodings** for both spatial and temporal context.
- **Transformer Encoder** for modeling complex relationships between stations and across time.
- **Fully Connected Layers** for generating final predictions.

This architecture enables **high-accuracy weather forecasts** by capturing **both temporal dependencies and geographical relationships**.

## Team Members

- **Yuval Rozner**
- **Dor Shabat**

## Tools and Technologies

- **Machine Learning & Data Processing**: Python, PyTorch, NumPy, Pandas, Scikit-learn
- **Frontend Development**: React, Material-UI, Styled-Components
- **Backend & Deployment**: Firebase Hosting, Firebase Functions, Git

## Contact

- **Yuval**: [yuvalrozner98@gmail.com](mailto:yuvalrozner98@gmail.com)
- **Dor**: [dorshabat55@gmail.com](mailto:dorshabat55@gmail.com)
