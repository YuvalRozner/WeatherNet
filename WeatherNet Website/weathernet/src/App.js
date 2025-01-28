import React, { useMemo } from "react";
import { AppProvider, DashboardLayout, PageContainer } from "@toolpad/core";
import { MyTheme } from "./utils/theme";
import {
  NavigationList,
  SidebarFooter,
} from "./utils/staticData/navigationList";
import { AppTitle, ToolbarActions } from "./components/topBarContent/topBar";
import { Home } from "./components/pages/home/home";
import {
  BrowserRouter,
  Routes,
  Route,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import { BaseLayout } from "./components/base/baseLayout";

// TODO: consider add skeltons components.
// TODO: consider add tooltips components.
// TODO: make the app responsive.
// TODO: add note for thanking the IMS.

/**
 * Builds and returns an array of <Route> elements, one per "segment"
 * in your NavigationList. (Skipping "divider"/"header" items.)
 */
function generateRouteElements(navList) {
  const routes = [];

  function recurse(prefix, items) {
    items.forEach((item) => {
      if (item.kind === "divider" || item.kind === "header") {
        return; // skip
      }
      if (item.segment) {
        let element;
        switch (item.segment) {
          case "":
            element = putInLayout(<Home />);
            break;
          default:
            element = putInLayout(item.pageComponent);
            if (!element) {
              element = putInLayout(<div>{item.segment} Page</div>);
            }
            break;
        }

        routes.push(
          <Route
            key={item.segment}
            path={`${prefix}/${item.segment}`}
            element={element}
          />
        );
      }
      // If there are children, recurse
      if (item.children && item.children.length > 0) {
        recurse(`${prefix}/${item.segment}`, item.children);
      }
    });
  }

  recurse("", navList);
  return routes;
}

/**
 * A simple custom hook that provides an object compatible with
 * the `AppProvider`'s 'router' prop.
 */
function useRouter() {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  return useMemo(
    () => ({
      pathname: location.pathname,
      searchParams,
      navigate: (path) => navigate(path),
    }),
    [location, searchParams, navigate]
  );
}

const putInLayout = (component) => {
  return (
    <>
      {component}
      <BaseLayout />
    </>
  );
};

function InnerApp() {
  const router = useRouter();

  // Generate all <Route> elements from NavigationList
  const dynamicRoutes = generateRouteElements(NavigationList);

  return (
    <AppProvider navigation={NavigationList} router={router} theme={MyTheme}>
      <DashboardLayout
        slots={{
          appTitle: AppTitle,
          sidebarFooter: SidebarFooter,
          toolbarActions: ToolbarActions,
        }}
      >
        <PageContainer
          sx={{
            maxWidth: "100% !important",
            margin: "0",
          }}
        >
          <Routes>
            <Route path="/" element={putInLayout(<Home />)} />
            {dynamicRoutes}
          </Routes>
        </PageContainer>
      </DashboardLayout>
    </AppProvider>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <InnerApp />
    </BrowserRouter>
  );
}
