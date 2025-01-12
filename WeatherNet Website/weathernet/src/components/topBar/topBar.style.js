import styled from 'styled-components';

export const TopBarContainer = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  display: flex;
  align-items: center;
  background-color: ${props => props.theme.background};
  color: ${props => props.theme.text};
  padding: 10px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  transition: background-color 0.3s ease;
`;

export const Logo = styled.img`
  height: 40px;
  margin-right: 22px;
`;

export const SiteName = styled.h1`
  color: white;
  font-size: 1.5rem;
  margin: 0;
`;
