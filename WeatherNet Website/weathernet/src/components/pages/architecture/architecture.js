import * as React from "react";
import { Typography, Container, Box } from "@mui/material";
import { content } from "../../../utils/staticData/aboutContent";

// TODO: change the content
export default function ArchitecturePage() {
  return (
    <Container>
      <Box my={4}>
        <Typography variant="h4" component="h1" gutterBottom>
          Hybrid Machine Learning Architecture
        </Typography>
        <Typography variant="body1" paragraph>
          {
            content.find(
              (item) => item.label === "Hybrid Machine Learning Architecture"
            ).description
          }
        </Typography>
        <Box mt={2}>
          <img
            src={
              content.find(
                (item) => item.label === "Hybrid Machine Learning Architecture"
              ).image
            }
            alt="Architecture"
            style={{ width: "100%", height: "auto" }}
          />
        </Box>
      </Box>
    </Container>
  );
}
