import { extendTheme } from "@mui/material";

export const MyTheme = extendTheme({
  colorSchemes: {
    light: true,
    dark: {
      palette: {
        background: {
          paper: "#282c34",
          default: "#282c34",
        },

        action: {
          hover: "rgba(41, 184, 200, 0.8)",
        },
      },
    },
  },
  colorSchemeSelector: "class",
});

export const lightTheme = {
  background: "#ffffff",
  text: "#000000",
  hoverText: "#000000b0",
  secondaryBackground: "grey",
  accent: "#29b8c8",
};

export const darkTheme = {
  background: "#282c34",
  text: "#ffffff",
  hoverText: "#29b8c8",
  secondaryBackground: "grey",
  accent: "#29b8c8",
};
