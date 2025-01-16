import React, { useState, useMemo } from "react";
import { AppProvider, DashboardLayout, PageContainer } from "@toolpad/core";
import { MyTheme } from "./utils/theme";
import { NavigationList } from "./utils/navigationList.js";
import { SidebarFooter } from "./utils/navigationList.js";
import { AppTitle, ToolbarActions } from "./components/topBarContent/topBar.js";
import { Home } from "./components/pages/home/home.js";

function App() {
  function useDemoRouter(initialPath) {
    const [pathname, setPathname] = useState(initialPath);

    const router = useMemo(
      () => ({
        pathname,
        searchParams: new URLSearchParams(),
        navigate: (path) => setPathname(String(path)),
      }),
      [pathname]
    );

    return router;
  }

  // Set up a demo router so the dashboard nav items can switch paths
  const router = useDemoRouter("/dashboard");

  return (
    <AppProvider navigation={NavigationList} router={router} theme={MyTheme}>
      <DashboardLayout
        slots={{
          appTitle: AppTitle,
          sidebarFooter: SidebarFooter,
          toolbarActions: ToolbarActions,
        }}
      >
        <PageContainer>
          <Home />
        </PageContainer>
      </DashboardLayout>
    </AppProvider>
  );
}

export default App;
