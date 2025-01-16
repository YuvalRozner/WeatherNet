import React from "react";

const Paper = ({ title, fileName }) => {
  return (
    <>
      <iframe
        src={fileName}
        title={title}
        width="100%"
        height="100%"
        style={{ border: "none" }}
      />
    </>
  );
};

export default Paper;
