import styled from 'styled-components';

export const ShareContainerWrapper = styled.div`
  display: flex;
  gap: 6px;
  justify-content: center;
  align-items: center;
  margin-top: 2px;
`;

export const ShareButton = styled.button`
  background-color: ${({ theme }) => theme.secondaryBackground}; 
  border: none;
  cursor: pointer;
  border-radius: 50%;
  width: 35px;
  height: 35px;
  display: flex;
  justify-content: center;
  align-items: center;
`;

export const ShareActions = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 4px;

  /* Transition effects */
  opacity: ${(props) => (props.isOpen ? '1' : '0')};
  visibility: ${(props) => (props.isOpen ? 'visible' : 'hidden')};

  /* Adjust translateX distance as preferred */
  transform: ${(props) => (props.isOpen ? 'translateX(0)' : 'translateX(40px)')};

  /* Tweak the transition durations/curves as needed */
  transition:
    opacity 0.3s ease-out,
    visibility 0s ease-in-out,
    transform 0.5s ease-in-out;
`;
