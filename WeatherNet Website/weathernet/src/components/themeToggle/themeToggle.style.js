import styled from 'styled-components';

export const ToggleContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 45px;
  height: 16px;
  background: darkgrey;
  border-radius: 16px;
  margin-left: 20px;
  margin-right: 50px;
  cursor: pointer;
`;

export const ThemeIcon = styled.div`
  color: ${(props) => props.theme.text};
  background: grey;
  transition: transform 0.7s ease;
  width: 28px;
  height: 28px;
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 40px;
  transform: ${(props) => props.$isVisible ? 'translateX(-5px)' : 'translateX(21px)'};

  &:hover {
    background: ${(props) => props.$isVisible ? 'lightgrey' : 'darkgrey'};
  }
`;
