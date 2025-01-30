import styled from "styled-components";
import Button from "@mui/material/Button";
import { BarChart } from "@mui/x-charts/BarChart";

export const Container = styled.div`
  /* Add any container-specific styles if needed */
`;

export const ImageContainer = styled.div`
  display: flex;
  margin-bottom: 16px;
  margin-top: 16px;
  align-items: center;
`;

export const Description = styled.p``;

export const HiddenImage = styled.img`
  display: none;
`;

export const StyledButton = styled(Button)`
  /* Add any additional styles for buttons if needed */
`;

export const CloseButton = styled(Button)`
  position: absolute;
  right: 10px;
  top: 10px;
  min-width: 2.1rem !important;
  padding: 0 !important;
  border-radius: 50% !important;

  span {
    font-size: 2.1rem;
    line-height: 1;
  }
`;

// Container that aligns all metric cards in one row, with horizontal scroll if needed:
export const CardsContainer = styled.div`
  display: flex;
  flex-wrap: nowrap; /* force single row */
  justify-content: space-evenly;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0;
  overflow-x: auto; // allow horizontal scroll if needed
`;

// Each metric card has a dynamic border color and consistent styling:
export const MetricCard = styled.div`
  flex: 0 0 auto; /* prevent card from growing/shrinking */
  width: 216px; /* fixed card width */
  border-radius: 6px;
  padding: 0.5rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border: 3px solid ${(props) => props.borderColor};
`;

// A new header container for the metric title and overall value:
export const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
`;

// Wrap the BarChart so it never spills out of the card:
export const ChartWrapper = styled.div`
  width: 100%;
  overflow: hidden; /* hide any chart overflow */
  box-sizing: border-box;
`;

// Title for each card:
export const MetricTitle = styled.h3`
  margin-top: 0;
  margin-bottom: 0.5rem;
`;

// Styled span for the overall value:
export const OverallValue = styled.span`
  font-size: 1.2rem;
  font-weight: bold;
`;

// Styled component to override BarChart bar styles
export const StyledBarChart = styled(BarChart)`
  & .MuiBarChart-bar {
    fill: white; /* Set bar fill to white */
    stroke: ${(props) =>
      props.borderColor}; /* Set bar border to card border color */
    stroke-width: 2px; /* Define bar border thickness */
  }
`;
