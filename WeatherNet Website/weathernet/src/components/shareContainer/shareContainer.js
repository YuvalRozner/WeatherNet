import React from 'react';
import {
  FacebookShareButton,
  TwitterShareButton,
  LinkedinShareButton,
  WhatsappShareButton,
  FacebookIcon,
  TwitterIcon,
  LinkedinIcon,
  WhatsappIcon
} from 'react-share';
import { ShareContainerWrapper } from './shareContainer.style';

const ShareContainer = ({ shareUrl, title }) => (
  <ShareContainerWrapper>
    <FacebookShareButton url={shareUrl} quote={title}>
      <FacebookIcon size={28} round />
    </FacebookShareButton>
    <TwitterShareButton url={shareUrl} title={title}>
      <TwitterIcon size={28} round />
    </TwitterShareButton>
    <LinkedinShareButton url={shareUrl} title={title}>
      <LinkedinIcon size={28} round />
    </LinkedinShareButton>
    <WhatsappShareButton url={shareUrl} title={title}>
      <WhatsappIcon size={28} round />
    </WhatsappShareButton>
  </ShareContainerWrapper>
);

export default ShareContainer; 