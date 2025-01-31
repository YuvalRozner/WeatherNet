import * as React from "react";
import { useState } from "react";
import { Container, Typography, Box, Skeleton } from "@mui/material";
import { useTheme } from "@mui/material/styles";

export default function ArchitecturePage() {
  const theme = useTheme();
  const themeMode = theme.palette.mode;
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <Container maxWidth="lg">
      <Box my={1}>
        <Typography variant="body1" paragraph>
          The <strong>WeatherNet Model</strong> we designed, predict weather
          parameters for a specific station by effectively capturing spatial and
          temporal dependencies from nearby stations.
        </Typography>

        <Typography variant="body1" paragraph>
          The architecture integrates{" "}
          <strong>1D Convolutional Neural Networks (CNNs)</strong> for feature
          extraction, positional encodings for spatial and temporal context, and
          a <strong>Transformer Encoder</strong> to model complex interactions
          between stations and across time windows. The final prediction layer
          outputs the desired weather metrics.
        </Typography>

        <Typography variant="h5" component="h2" gutterBottom>
          <strong>Hybrid Model Approach</strong>
        </Typography>

        <Typography variant="body1" paragraph>
          We employ a multi-model strategy, utilizing four instances of the
          <strong> WeatherNet Model</strong>, each optimized for different
          forecasting horizons. While all models share the same architecture,
          they specialize in distinct time frames to enhance predictive
          accuracy:
        </Typography>

        <ul>
          <li>
            <Typography variant="body1">
              <strong>0-12 hours:</strong> Captures short-term weather dynamics.
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>12-24 hours:</strong> Provides insights into near-daily
              trends.
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>24-36 hours:</strong> Focuses on mid-term patterns.
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>36-60 hours:</strong> Optimized for longer-range
              forecasts.
            </Typography>
          </li>
        </ul>

        <Typography variant="body1" paragraph>
          By training each model on its respective time window, we ensure that
          learned representations are tailored to specific temporal
          dependencies, improving accuracy across different forecasting
          horizons.
        </Typography>

        <Typography variant="h5" component="h2" gutterBottom>
          <strong>WeatherNet Architecture Diagram</strong>
        </Typography>

        {/* <Typography variant="h5" component="h2" gutterBottom>
          <strong>Conclusion</strong>
        </Typography>

        <Typography variant="body1" paragraph>
          The <strong>WeatherNetModel</strong> integrates CNN-based feature
          extraction, positional encodings, and Transformer-based encoding to
          deliver precise weather predictions for targeted stations. Its modular
          design ensures flexibility and scalability, making it well-suited for
          diverse meteorological applications. By leveraging both spatial and
          temporal contexts, the model significantly enhances predictive
          performance in complex weather systems.
        </Typography> */}

        <Box my={4} display="flex" justifyContent="center">
          {!imageLoaded && (
            <Skeleton
              variant="rectangular"
              width="100%"
              height={400}
              sx={{ borderRadius: "10px" }}
            />
          )}
          <img
            src={`/figures/architecture_${themeMode}.png`}
            alt="Architecture Diagram"
            style={{
              maxWidth: "100%",
              height: "auto",
              display: imageLoaded ? "block" : "none",
            }}
            onLoad={() => setImageLoaded(true)}
          />
        </Box>
        <Typography variant="h5" component="h2" gutterBottom>
          <strong>Detailed Component Analysis</strong>
        </Typography>

        <Typography variant="h6" component="h3" gutterBottom>
          <u>1D Convolutional Neural Network (CNN)</u>
        </Typography>
        <Typography variant="body1" paragraph>
          The <strong>1D CNN</strong> module is responsible for extracting
          high-level temporal features from the raw input data of each weather
          station. By applying convolutional operations, it captures local
          temporal patterns and trends essential for accurate predictions. It
          works on each feature independently.
        </Typography>
        <Typography variant="body1" paragraph>
          Each weather station is assigned an individual
          <strong> 1D CNN</strong> instance.
        </Typography>

        <Typography variant="h6" component="h3" gutterBottom>
          <u>
            Positional Encoding - Geographic -{" "}
            <i>Coordinate Positional Encoding</i>
          </u>
        </Typography>
        <Typography variant="body1" paragraph>
          Spatial context is vital in weather prediction as geographical
          proximity often correlates with similar weather patterns.
          <br />
          The{" "}
          <strong>
            <i>Coordinate Positional Encoding </i>
          </strong>
          module encodes the spatial coordinates (East, North in Israeli
          Transverse Mercator) of each station into a fixed-dimensional
          embedding, enriching the model with geographical information.
        </Typography>

        <Typography variant="h6" component="h3" gutterBottom>
          <u>
            Positional Encoding - Time - <i>Temporal Positional Encoding</i>
          </u>
        </Typography>
        <Typography variant="body1" paragraph>
          Temporal dependencies are inherent in weather data, where past
          patterns influence future conditions. The{" "}
          <strong>
            <i>Temporal Positional Encoding</i>
          </strong>{" "}
          module injects information about the position of each time step into
          the model, allowing it to recognize the sequence and timing of events.
        </Typography>
        <Typography variant="body1" paragraph>
          <strong>Sinusoidal Positional Encodings:</strong> Generates fixed
          positional encodings using sine and cosine functions of varying
          frequencies, as introduced in the original Transformer architecture.
        </Typography>

        <Typography variant="h6" component="h3" gutterBottom>
          <u>Transformer Encoder</u>
        </Typography>
        <Typography variant="body1" paragraph>
          The <strong>Transformer Encoder</strong> excels at modeling complex
          dependencies and interactions within sequential data.
          <br /> In our architecture, it processes the combined spatial-temporal
          features to capture intricate patterns that influence weather
          predictions.
        </Typography>

        <Typography variant="h6" component="h3" gutterBottom>
          <u>Fully Connected Layer (FC)</u>
        </Typography>
        <Typography variant="body1" paragraph>
          To bridge the gap between high-dimensional Transformer outputs and the
          desired prediction space, a linear layer maps the aggregated features
          to the target output dimensions.
        </Typography>
        <Typography variant="body1" paragraph>
          This final layer translates the rich feature representations into
          actionable predictions, series of temperatures (size based on
          parameter) for the target station.
        </Typography>
      </Box>
    </Container>
  );
}
