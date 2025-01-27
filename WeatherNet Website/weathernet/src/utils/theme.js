import { extendTheme } from "@mui/material";

export const MyTheme = extendTheme({
  colorSchemes: {
    // Provide an object for `light` if needed
    light: {
      // e.g. palette: { ... }
    },
    dark: {
      palette: {
        background: {
          paper: "#282c34",
          default: "#282c34",
        },
        action: {
          hover: "rgba(41, 184, 200, 0.7)",
          helpButtonHover: "rgba(255, 255, 255, 0.9)",
          disabled: "rgba(255, 255, 255, 0.3)",
        },
      },
    },
  },
  colorSchemeSelector: "class",
  components: {
    MuiSvgIcon: {
      styleOverrides: {
        root: {
          // Default size for all icons
          width: "1.1em",
          height: "1.1em",
          // Specific sizes for MenuOpenIcon and MenuIcon
          "@media (min-width: 600px)": {
            '&[data-testid="MenuOpenIcon"], &[data-testid="MenuIcon"]': {
              width: "1.4em",
              height: "1.7em",
            },
          },
        },
      },
    },
    MuiSpeedDial: {
      defaultProps: {
        FabProps: { size: "small" },
      },
    },
    MuiFab: {
      styleOverrides: {
        sizeSmall: {
          width: "40px",
          height: "40px",
          minHeight: "40px",
          padding: "4px",
          "@media (max-width:600px)": {
            width: "31px",
            height: "31px",
            minHeight: "31px",
            padding: "3px",
          },
        },
      },
    },
  },
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 900,
      lg: 1200,
      xl: 1536,
    },
  },
});
