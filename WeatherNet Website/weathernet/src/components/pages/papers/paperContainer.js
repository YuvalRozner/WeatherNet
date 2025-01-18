import React, { useState, useEffect, useMemo } from "react";
import Paper from "./paper";
import { NavigationList } from "../../../utils/navigationList";
import { GridBox } from "./paper.style";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import CardActionArea from "@mui/material/CardActionArea";

// TODO: fix routing in navigationbar.
export const PaperContainer = ({ id, title, fileName }) => {
  const [selectedPaper, setSelectedPaper] = useState({ id, title, fileName });
  const [selectedCard, setSelectedCard] = useState(null);

  // Find the segment in NavigationList
  const papersAndManuals = NavigationList.find(
    (item) => item.segment === "PapersAndManuals"
  );

  const cards = useMemo(() => {
    return (
      papersAndManuals?.children?.map((child, index) => ({
        id: index,
        title: child.title,
        fileName: child.fileName,
        description: `${child.title}.`, // TODO: Add description
      })) || []
    );
  }, [papersAndManuals]);

  useEffect(() => {
    if (selectedCard !== null) {
      setSelectedPaper(cards[selectedCard]);
    }
  }, [selectedCard, cards]);

  const handleCardClick = (index) => {
    setSelectedCard(index);
  };

  return (
    <>
      <GridBox>
        {cards.map((card, index) => (
          <Card key={card.id}>
            <CardActionArea
              onClick={() => handleCardClick(index)}
              data-active={selectedCard === index ? "" : undefined}
              sx={{
                height: "100%",
                "&[data-active]": {
                  backgroundColor: "action.selected",
                  "&:hover": {
                    backgroundColor: "action.selectedHover",
                  },
                },
              }}
            >
              <CardContent sx={{ height: "100%" }}>
                <Typography variant="h5" component="div">
                  {card.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {card.description}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        ))}
      </GridBox>

      {selectedPaper && (
        <Paper title={selectedPaper.title} fileName={selectedPaper.fileName} />
      )}
    </>
  );
};

export default PaperContainer;
