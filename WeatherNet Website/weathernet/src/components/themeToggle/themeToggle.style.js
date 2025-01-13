// ToggleStyles.js

import styled from 'styled-components';

export const ToggleContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 52px; /* or some fixed width */
  background: ${(props) => 'darkgrey'};
  border-radius: 16px;
  padding: 4px 5px;
  margin-left: 20px;
  margin-right: 50px;
  cursor: pointer;
`;

export const DarkIcon = styled.div`
  color: ${(props) => props.theme.text};
  background: ${(props) => props.isVisible ? 'grey' : 'transparent'};
  transition: color 0.4s ease, background 0.8s ease;
  width: 24px;
  height: 24px;
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 16px;

  &:hover {
    color: ${(props) => props.theme.hoverText};
  }
`;

export const LightIcon = styled.div`
  color: ${(props) => props.theme.text};
  transition: color 0.4s ease, background 0.8s ease;
  background: ${(props) => props.isVisible ? 'grey' : 'transparent'};
  width: 24px;
  height: 24px;
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 16px;

  &:hover {
    color: ${(props) => props.theme.hoverText};
  }
`;
