import React, { useState, useEffect, useMemo } from "react";
import { NavigationList } from "../../../utils/navigationList";
import { GridBox } from "./paper.style";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import CardActionArea from "@mui/material/CardActionArea";
import { useNavigate } from "react-router-dom";
import Tooltip from "@mui/material/Tooltip";

const FilesPicker = ({ onSelectPaper, selectedCardId }) => {
  const [selectedCard, setSelectedCard] = useState(selectedCardId);
  const navigate = useNavigate();

  const handleCardClick = (index, selectedSegment) => {
    setSelectedCard(index);
    navigate(`/PapersAndManuals/${selectedSegment}`); // Navigate to the segment
  };

  // Find the segment in NavigationList
  const papersAndManuals = NavigationList.find(
    (item) => item.segment === "PapersAndManuals"
  );

  const cards = useMemo(() => {
    return (
      papersAndManuals?.children?.map((child, index) => ({
        id: index,
        segment: child.segment,
        title: child.title,
        fileName: child.fileName,
        description: child.description,
      })) || []
    );
  }, [papersAndManuals]);

  useEffect(() => {
    if (selectedCard !== null && onSelectPaper) {
      onSelectPaper(cards[selectedCard]);
    }
  }, [selectedCard, cards, onSelectPaper]);

  return (
    <GridBox $columns={cards.length}>
      {cards.map((card, index) => (
        <Tooltip title="Switch file" arrow placement="top-end">
          <Card key={card.id}>
            <CardActionArea
              onClick={() => handleCardClick(index, card.segment)}
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
        </Tooltip>
      ))}
    </GridBox>
  );
};

export default FilesPicker;
