import React, { useState } from "react";
import Paper from "./paper";
import FilesPicker from "./filesPicker";

// TODO: fix routing in navigationbar.
export const PaperContainer = ({ index, title, fileName }) => {
  const [selectedPaper, setSelectedPaper] = useState({
    index,
    title,
    fileName,
  });

  return (
    <>
      <FilesPicker
        onSelectPaper={setSelectedPaper}
        selectedCardId={selectedPaper.index}
      />
      {selectedPaper && (
        <Paper title={selectedPaper.title} fileName={selectedPaper.fileName} />
      )}
    </>
  );
};

export default PaperContainer;
