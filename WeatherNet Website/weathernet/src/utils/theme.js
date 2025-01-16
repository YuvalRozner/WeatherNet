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
