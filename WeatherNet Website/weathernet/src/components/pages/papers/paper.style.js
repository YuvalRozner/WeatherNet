import styled from "styled-components";

export const GridBox = styled.div`
  width: 100%;
  display: grid;
  justify-content: center;
  grid-template-columns: ${({ $columns }) => `repeat(${$columns}, 236px)`};
  gap: 15px;
`;
