import { Typography } from "@mui/material";
import {
  Description,
  HomeOutlined,
  ThermostatOutlined,
  BarChartOutlined,
  AccountBoxOutlined,
  ContactSupportOutlined,
  IntegrationInstructionsOutlined,
  AssignmentIndOutlined,
  InsightsOutlined,
  NotesOutlined,
  FileCopyOutlined,
} from "@mui/icons-material";
import { Home } from "../components/pages/home/home.js";
import Paper from "../components/pages/papers/paper.js";
import About from "../components/pages/about/about.js";
import FilesContainer from "../components/pages/papers/filesContainer.js";
import Contributors from "../components/pages/contributers/contributors.js";

export const NavigationList = [
  {
    segment: "Home",
    title: "Home",
    icon: <HomeOutlined />,
    pageComponent: <Home />,
  },
  {
    kind: "divider",
  },
  {
    kind: "header",
    title: "Weather",
  },
  {
    segment: "Forecasts",
    title: "Forecasts",
    icon: <ThermostatOutlined />,
    children: [
      {
        segment: "WeatherNet's",
        title: "WeatherNet's",
        icon: (
          <img
            src={"/logo/weathernet_small_icon.png"}
            alt="weatherNet small logo"
            style={{ height: "38px" }}
          />
        ),
      },
      {
        segment: "IMS's",
        title: "IMS's",
        icon: (
          <img
            src={"/logo/ims_small_icon.jpg"}
            alt="IMS's small logo"
            style={{ height: "30px", margin: "0px 6px" }}
          />
        ),
      },
    ],
  },
  {
    segment: "Statistics",
    title: "Statistics",
    icon: <BarChartOutlined />,
  },
  {
    kind: "divider",
  },
  {
    kind: "header",
    title: "Model",
  },
  {
    segment: "Architecture",
    title: "Architecture",
    icon: <InsightsOutlined />,
  },
  {
    segment: "AboutWeatherNet",
    title: "About WeatherNet",
    icon: <NotesOutlined />,
    pageComponent: <About />,
  },
  {
    segment: "PapersAndManuals",
    title: "Papers and Manuals",
    icon: <FileCopyOutlined />,
    pageComponent: <FilesContainer />,
    children: [
      {
        segment: "PaperPhaseA",
        title: "Paper Phase A",
        icon: <Description />,
        pageComponent: (
          <Paper
            title="Paper Phase A"
            fileName="/papers/WeatherNet - Phase A Paper.pdf"
          />
        ),
      },
      {
        segment: "PaperPhaseB",
        title: "Paper Phase B",
        icon: <Description />,
        pageComponent: (
          <Paper
            title="Paper Phase A"
            fileName="/papers/WeatherNet - Phase A Paper.pdf"
          />
        ),
      },
      {
        segment: "UserManual",
        title: "User Manual",
        icon: <AssignmentIndOutlined />,
        pageComponent: (
          <Paper
            title="Paper Phase A"
            fileName="/papers/WeatherNet - Phase A Paper.pdf"
          />
        ),
      },
      {
        segment: "DeveloperManual",
        title: "Developer Manual",
        icon: <IntegrationInstructionsOutlined />,
        pageComponent: (
          <Paper
            title="Developer Manual"
            fileName="/papers/WeatherNet - Phase A Paper.pdf"
          />
        ),
      },
    ],
  },
  {
    kind: "divider",
  },
  {
    kind: "header",
    title: "Who are We?",
  },
  {
    segment: "Contributors",
    title: "Contributors",
    icon: <AccountBoxOutlined />,
    pageComponent: <Contributors />,
  },
  {
    segment: "ContactUs",
    title: "Contact us",
    icon: <ContactSupportOutlined />,
  },
];

export function SidebarFooter({ mini }) {
  return (
    <Typography
      variant="caption"
      sx={{ m: 1, whiteSpace: "nowrap", overflow: "hidden" }}
    >
      {mini
        ? "© WeatherNet"
        : `© ${new Date().getFullYear()} Made by Dor and Yuval`}
    </Typography>
  );
}
