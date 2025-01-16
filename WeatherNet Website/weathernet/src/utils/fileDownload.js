export function downloadFile(fileName) {
  const link = document.createElement("a");
  link.href = fileName;
  link.download = fileName.split("/").pop();
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
