import React from "react";
import Skeleton from "@mui/material/Skeleton";
import Box from "@mui/material/Box";

export const SkeletonLayout = () => {
  return (
    <Box sx={{ width: "100%", padding: 2 }}>
      <Skeleton variant="text" width="60%" height={40} />
      <Skeleton
        variant="rectangular"
        width="100%"
        height={200}
        sx={{ my: 2 }}
      />
      <Skeleton variant="text" width="80%" height={30} />
      <Skeleton variant="text" width="80%" height={30} />
      <Skeleton variant="text" width="80%" height={30} />
      <Skeleton
        variant="rectangular"
        width="100%"
        height={400}
        sx={{ my: 2 }}
      />
    </Box>
  );
};
