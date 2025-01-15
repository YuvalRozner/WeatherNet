import React, { useState } from 'react';
import {
  FacebookShareButton,
  TwitterShareButton,
  LinkedinShareButton,
  WhatsappShareButton,
  FacebookIcon,
  TwitterIcon,
  LinkedinIcon,
  WhatsappIcon,
} from 'react-share';
import { ShareContainerWrapper, ShareButton, ShareActions } from './shareContainer.style';

const ShareContainer = ({ shareUrl, title, theme }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [buttonHovered, setButtonHovered] = useState(false);

  const toggleShareOptions = () => {
    setIsOpen(!isOpen);
  };

  const actions = [
    {
      name: 'Facebook',
      button: FacebookShareButton,
      icon: FacebookIcon,
    },
    {
      name: 'Twitter',
      button: TwitterShareButton,
      icon: TwitterIcon,
    },
    {
      name: 'LinkedIn',
      button: LinkedinShareButton,
      icon: LinkedinIcon,
    },
    {
      name: 'WhatsApp',
      button: WhatsappShareButton,
      icon: WhatsappIcon,
    },
  ];

  return (
    <ShareContainerWrapper>
      
      <ShareActions isOpen={isOpen || buttonHovered}>
        {actions.map((action) => {
          const ButtonComponent = action.button;
          const IconComponent = action.icon;
          return (
            <ButtonComponent
              key={action.name}
              url={shareUrl}
              title={title}
              className="share-action"
            >
              <IconComponent size={28} round />
            </ButtonComponent>
          );
        })}
      </ShareActions>

      <ShareButton onClick={toggleShareOptions} className="share-button">
        <i 
          className="fi fi-rr-share" 
          style={{ 
            color: 'white', 
            fontSize: '18px', 
            marginTop: '5px', 
            marginRight: '2px',
            transition: 'color 0.2s ease',
          }} 
          onMouseEnter={(e) => {
            e.currentTarget.style.color = '#29b8c8';
            setButtonHovered(!buttonHovered);
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'white';
            setButtonHovered(!buttonHovered);
          }}
        />
      </ShareButton>
    </ShareContainerWrapper>
  );
};

export default ShareContainer;
