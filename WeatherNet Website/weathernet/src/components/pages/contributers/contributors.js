import * as React from "react";
import {
  ProfilesCardsContainer,
  NameContainer,
  AboutContainer,
  RoleContainer,
  IconContainer,
  StyledDialogContentText,
  StyledDialogContent,
  StyledDialogTitle,
  StyledCard,
} from "./contributors.style.js";
import {
  CardContent,
  CardMedia,
  CardActionArea,
  Dialog,
  DialogActions,
  Button,
} from "@mui/material";
import {
  Email as EmailIcon,
  LinkedIn as LinkedInIcon,
  GitHub as GitHubIcon,
  Phone as PhoneIcon,
} from "@mui/icons-material";
import { ContributorsList } from "../../../utils/staticData/contributorsList.js";
import { DraggablePaperComponent } from "../../../utils/logicForComponents/draggable.js";
import Tooltip from "@mui/material/Tooltip";

export default function Contributors() {
  const [open, setOpen] = React.useState(false);
  const [selectedPerson, setSelectedPerson] = React.useState(null);

  const handleClickOpen = (person) => {
    setSelectedPerson(person);
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setSelectedPerson(null);
  };

  return (
    <ProfilesCardsContainer>
      {ContributorsList.map((person, index) => (
        <StyledCard key={index}>
          <Tooltip
            title="Click to view contact details"
            arrow
            placement="bottom-end"
          >
            <CardActionArea onClick={() => handleClickOpen(person)}>
              <CardMedia
                component="img"
                height="190"
                image={person.image}
                alt={person.name}
              />
              <CardContent>
                <NameContainer>{person.name}</NameContainer>
                <AboutContainer>{person.about}</AboutContainer>
                <RoleContainer>{person.role}</RoleContainer>
              </CardContent>
            </CardActionArea>
          </Tooltip>
        </StyledCard>
      ))}

      <Dialog
        open={open}
        onClose={handleClose}
        PaperComponent={DraggablePaperComponent}
        aria-labelledby="draggable-dialog-title"
      >
        <StyledDialogTitle id="draggable-dialog-title">
          Contact Details
        </StyledDialogTitle>
        <StyledDialogContent>
          {selectedPerson && (
            <>
              <StyledDialogContentText>
                <IconContainer>
                  <EmailIcon />
                </IconContainer>
                <strong>Email:</strong>{" "}
                <a href={`mailto:${selectedPerson.email}`}>
                  {selectedPerson.email}
                </a>
              </StyledDialogContentText>
              <StyledDialogContentText>
                <IconContainer>
                  <LinkedInIcon />
                </IconContainer>
                <strong>LinkedIn:</strong>{" "}
                <a
                  href={selectedPerson.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {selectedPerson.name}'s LinkedIn
                </a>
              </StyledDialogContentText>
              <StyledDialogContentText>
                <IconContainer>
                  <GitHubIcon />
                </IconContainer>
                <strong>GitHub:</strong>{" "}
                <a
                  href={selectedPerson.github}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {selectedPerson.name}'s GitHub
                </a>
              </StyledDialogContentText>
              <StyledDialogContentText>
                <IconContainer>
                  <PhoneIcon />
                </IconContainer>
                <strong>Phone:</strong> {selectedPerson.phone}
              </StyledDialogContentText>
            </>
          )}
        </StyledDialogContent>
        <DialogActions>
          <Button onClick={handleClose} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </ProfilesCardsContainer>
  );
}
