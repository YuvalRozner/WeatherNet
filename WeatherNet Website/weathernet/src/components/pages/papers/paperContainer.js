import React, { useState } from "react";
import Paper from "./paper";
import FilesPicker from "./filesPicker";

// TODO: fix routing in navigationbar.
export const PaperContainer = ({ id, title, fileName }) => {
  const [selectedPaper, setSelectedPaper] = useState({ id, title, fileName });

  return (
    <>
      <FilesPicker onSelectPaper={setSelectedPaper} />
      {selectedPaper && (
        <Paper title={selectedPaper.title} fileName={selectedPaper.fileName} />
      )}
    </>
  );
};

export default PaperContainer;
