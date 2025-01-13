import styled from 'styled-components';

export const TopBarContainer = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: ${props => props.theme.background};
  color: ${props => props.theme.text};
  padding: 0 20px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  transition: background-color 0.3s ease;
`;

export const Logo = styled.img`
  height: 44px;
  margin: 0px 22px 0px 6px;
`;

export const SiteName = styled.h1`
  color: ${props => props.theme.text};
  font-size: 1.5rem;
  margin: 0;

  &:hover {
    color: ${props => props.theme.hoverText};
  }
`;

export const ShareButtonsContainer = styled.div`
  display: flex;
  align-items: center;
  margin-left: auto;
`;
