# PKL Viewer

PKL Viewer is a simple GUI application for viewing the contents of Pickle (.pkl) files. It supports loading the entire file at once or loading the file in chunks to avoid blocking the main thread.

# Modes
## Full load
In this mode, the entire file is loaded into memory and displayed in the GUI.
it is faster if we want to load the whole file. but untill it is not loading the Gui is not responsive.

## Load In Chuncks
In this mode, the file is loaded in chunks and displayed in the GUI. This mode is slower than the full load mode but the GUI remains responsive.
there is option to pause the loading (the pkl is loaded to memory but not displayed) and resume the loading.

# Recommendations
in general if we want to see somthink quickly we should use the **'Load In Chuncks'** mode. 

# Future Work
- Add support for loading the file in chunks in the background.
- Add scrolling ui for the loaded data.
- Add support for sort by clicking on the column header.
- Changing the buttons layout to make it more user friendly.