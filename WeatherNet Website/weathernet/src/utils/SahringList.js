import { FacebookIcon, XIcon, LinkedinIcon, WhatsappIcon } from "react-share";
import LaunchRoundedIcon from "@mui/icons-material/LaunchRounded";

export const sharingOptions = [
  {
    name: "LinkedIn",
    IconComponent: LinkedinIcon,
    url: (shareUrl) =>
      `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(
        shareUrl
      )}`,
    windowName: "linkedin-share-dialog",
  },
  {
    name: "WhatsApp",
    IconComponent: WhatsappIcon,
    url: (shareUrl, title) =>
      `https://api.whatsapp.com/send?text=${encodeURIComponent(
        title
      )}%20${encodeURIComponent(shareUrl)}`,
    windowName: "whatsapp-share-dialog",
  },
  {
    name: "X",
    IconComponent: XIcon,
    url: (shareUrl, title) =>
      `https://x.com/intent/tweet?text=${encodeURIComponent(
        title
      )}&url=${encodeURIComponent(shareUrl)}`,
    windowName: "x-share-dialog",
  },
  {
    name: "Facebook",
    IconComponent: FacebookIcon,
    url: (shareUrl) =>
      `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(
        shareUrl
      )}`,
    windowName: "facebook-share-dialog",
  },
  {
    name: "Copy Link",
    IconComponent: LaunchRoundedIcon,
    action: (shareUrl) => {
      navigator.clipboard.writeText(shareUrl);
    },
  },
];
