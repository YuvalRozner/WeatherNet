import styled from "styled-components";

export const SiteNameContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const SiteName = styled.h1`
  color: ${(props) => props.theme.text};
  font-size: 1.7rem;
  margin: 0;

  &:hover {
    color: ${(props) => props.theme.hoverText};
  }
`;

export const Logo = styled.img`
  height: 60px;
  margin: 0px 22px 0px 6px;
`;

export const InvisibleDiv = styled.div`
  width: 54px;
`;

export const ShareButtonsContainer = styled.div`
  display: flex;
  align-items: center;
  margin-left: auto;
`;
