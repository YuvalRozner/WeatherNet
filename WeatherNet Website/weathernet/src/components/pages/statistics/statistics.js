import { Container } from "./statistics.style";
import React from "react";
import StatisticsGraphsContainer from "./statisticsGraphsContainer";
import ModelMetrics from "./modelMetrics";

export default function Statistics() {
  return (
    <Container>
      <ModelMetrics />
      <StatisticsGraphsContainer />
    </Container>
  );
}
