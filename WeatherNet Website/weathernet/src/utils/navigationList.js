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
import Home from "../components/pages/home/home.js";
import PaperContainer from "../components/pages/papers/paperContainer.js";
import About from "../components/pages/about/about.js";
import Contributors from "../components/pages/contributers/contributors.js";
import ContactUs from "../components/pages/contactUs/contactUs.js";
import Architecture from "../components/pages/architecture/architecture.js";
import Statistics from "../components/pages/statistics/statistics.js";

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
    pageComponent: <Statistics />,
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
    pageComponent: <Architecture />,
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
    pageComponent: <PaperContainer key="PapersAndManuals" index={0} />,
    children: [
      {
        segment: "PaperPhaseA",
        title: "Paper Phase A",
        icon: <Description />,
        fileName: "/papers/WeatherNet - Phase A Paper.pdf",
        pageComponent: (
          <PaperContainer
            key="PaperPhaseA"
            index={0}
            title="PaperPhaseA"
            fileName="/papers/WeatherNet - Phase A Paper.pdf"
          />
        ),
      },
      {
        segment: "PaperPhaseB",
        title: "Paper Phase B",
        icon: <Description />,
        fileName: "/papers/WeatherNet - Phase B Paper.pdf", // TODO: update the file Phase B Paper
        pageComponent: (
          <PaperContainer
            key="PaperPhaseB"
            index={1}
            title="PaperPhaseB"
            fileName="/papers/WeatherNet - Phase B Paper.pdf"
          />
        ),
      },
      {
        segment: "UserManual",
        title: "User Manual",
        icon: <AssignmentIndOutlined />,
        fileName: "/papers/WeatherNet - User Manual.pdf", // TODO: update the file User Manual
        pageComponent: (
          <PaperContainer
            key="UserManual"
            index={2}
            title="UserManual"
            fileName="/papers/WeatherNet - User Manual.pdf"
          />
        ),
      },
      {
        segment: "DeveloperManual",
        title: "Developer Manual",
        icon: <IntegrationInstructionsOutlined />,
        fileName: "/papers/WeatherNet - Developer Manual.pdf", // TODO: update the file Developer Manual
        pageComponent: (
          <PaperContainer
            key="DeveloperManual"
            index={3}
            title="DeveloperManual"
            fileName="/papers/WeatherNet - Developer Manual.pdf"
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
    pageComponent: <ContactUs />,
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
