import React, { useState } from "react";
import { ResizableBox } from "react-resizable";
import "react-resizable/css/styles.css";

const Paper = ({ title, fileName }) => {
  const [size, setSize] = useState({ width: "86%", height: 700 });

  const handleResize = (event, { size: newSize }) => {
    setSize(newSize);
  };

  return (
    <ResizableBox
      width={size.width}
      height={size.height}
      minConstraints={[300, 300]}
      maxConstraints={[1200, 1200]}
      onResize={handleResize}
      resizeHandles={["se", "sw"]}
      style={{ margin: "13px 3%" }}
    >
      <iframe
        src={fileName}
        title={title}
        width="100%"
        height="100%"
        style={{ border: "none", borderRadius: "12px" }}
      />
    </ResizableBox>
  );
};

export default Paper;
