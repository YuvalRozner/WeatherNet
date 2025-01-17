import styled from "styled-components";
import {
  DialogContentText,
  DialogContent,
  DialogTitle,
  Card,
} from "@mui/material";

export const ProfilesCardsContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 40px;
  margin-top: 40px;
`;

export const NameContainer = styled.div`
  font-weight: 300;
  font-size: 1.7rem;
  margin-bottom: 10px;
`;

export const AboutContainer = styled.div`
  font-size: 1rem;
  margin-bottom: 5px;
`;

export const RoleContainer = styled.div`
  font-size: 1rem;
`;

export const IconContainer = styled.span`
  vertical-align: middle;
  margin-right: 12px;
  display: inline-block;
`;

export const StyledDialogContentText = styled(DialogContentText)`
  margin-bottom: 6px;
`;

export const StyledDialogContent = styled(DialogContent)`
  padding: 8px 50px;
`;

export const StyledDialogTitle = styled(DialogTitle)`
  cursor: move;
`;

export const StyledCard = styled(Card)`
  width: 400px;
  margin-bottom: 2px;
`;
