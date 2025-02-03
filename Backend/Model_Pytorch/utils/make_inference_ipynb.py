# pip install nbformat
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import re
import os

def remove_code_above_main(code):
    """
    Removes all lines above the 'if __name__ == "__main__":' statement.
    
    Parameters:
    - code (str): The original code.
    
    Returns:
    - str: The code with lines above 'def main()::' removed.
    """
    lines = code.splitlines()
    new_lines = []
    main_found = False
    
    for idx, line in enumerate(lines):
        if 'def main():' in line:
            main_found = True
            # Optionally include the 'if __name__ == "__main__":' line
            new_lines.append(line)
            # If you don't want to keep it, comment out the above line
            continue
        if main_found:
            new_lines.append(line)
    
    if not main_found:
        print("Warning: 'if __name__ == \"__main__\":' not found in the code.")
        return code  # Return the original code if the main block isn't found
    
    return "\n".join(new_lines)


def apply_alterations(code, alterations):
    """
    Applies a list of alteration rules to the given code.

    Parameters:
    - code (str): The original code.
    - alterations (list of tuples): Each tuple contains a pattern and its replacement.

    Returns:
    - str: The altered code.
    """
    for pattern, replacement in alterations:
        code = re.sub(pattern, replacement, code)
    return code

def add_code_block_to_start(nb, code_str, markdown=None):
    """
    Adds a code block to the beginning of the given notebook from a string.
    
    Parameters:
    - nb (nbformat.notebooknode.NotebookNode): The notebook object to prepend the code cell to.
    - code_str (str): The code string to add as a code cell.
    - markdown (str, optional): Optional markdown text to add before the code cell.
    """
    # List to hold the new cells to be prepended
    new_cells = []
    
    # Optionally add a markdown cell before the code cell
    if markdown:
        markdown_cell = nbformat.v4.new_markdown_cell(markdown)
        new_cells.append(markdown_cell)
    
    # Create the new code cell
    code_cell = new_code_cell(code_str)
    new_cells.append(code_cell)
    
    # Prepend the new cells to the existing notebook cells
    nb.cells = new_cells + nb.cells

def create_notebook_from_scripts(file_paths, output_notebook, alteration_rules=None, initial_code_blocks=None):
    """
    Creates a Jupyter Notebook with each Python file's content in separate code cells,
    applying any specified alterations, and optionally adding initial code blocks at the start.
    
    Parameters:
    - file_paths (list of str): List of paths to the Python (.py) files.
    - output_notebook (str): Path where the output .ipynb file will be saved.
    - alteration_rules (dict): Optional. A dictionary mapping file paths to their
      respective alteration rules.
    - initial_code_blocks (list of tuples): Optional. Each tuple contains optional markdown and code strings to add at the start.
    """
    # Create a new notebook object
    nb = new_notebook()

    # Add initial code blocks if any
    if initial_code_blocks:
        for markdown, code in reversed(initial_code_blocks):
            add_code_block_to_start(nb, code, markdown)

    # Mapping of function names to actual functions
    function_mapping = {
        "remove_code_above_main": remove_code_above_main,
        # Add more function mappings here if you introduce additional functions
    }

    # Iterate over each file path
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Apply alterations if any are specified for this file
            if alteration_rules and path in alteration_rules:
                # Apply regex-based alterations
                regex_rules = alteration_rules[path].get("regex", [])
                for pattern, replacement in regex_rules:
                    code = re.sub(pattern, replacement, code)

                # Apply function-based alterations
                function_rules = alteration_rules[path].get("functions", [])
                for func_name in function_rules:
                    func = function_mapping.get(func_name)
                    if func:
                        code = func(code)
                    else:
                        print(f"Warning: Function '{func_name}' is not defined.")

            # Optionally, add a markdown cell with the file name
            markdown_cell = f"### `{os.path.basename(path)}`"
            nb.cells.append(nbformat.v4.new_markdown_cell(markdown_cell))

            # Create a new code cell with the (altered) file's content
            code_cell = new_code_cell(code)
            nb.cells.append(code_cell)

        except FileNotFoundError:
            print(f"Error: The file {path} was not found.")
        except Exception as e:
            print(f"An error occurred while processing {path}: {e}")

    # Write the notebook object to a file
    try:
        with open(output_notebook, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Notebook successfully created at {output_notebook}")
    except Exception as e:
        print(f"Failed to write the notebook file: {e}")

if __name__ == "__main__":
    path_to_dir = os.path.dirname(os.path.abspath(__file__))

    # Define your initial code and markdown
    markdown_1 = "### numpy"

    code_1 = """
!pip uninstall numpy -y
!pip install numpy
!pip install --upgrade pandas
"""
    code_2 = """
!pip install --upgrade --force-reinstall numpy pandas
"""
    code_3 = """
import numpy as np
print(np.__version__)
"""
    colab_mount = """### Colab Mount"""
    code_4 = """
from google.colab import drive
drive.mount('/content/drive')
"""
    unzip_weights = """### Unzip Weights"""
    code_5 = """
import zipfile
zip_file_path = "/content/models_for_inference.zip"

# Extract the zip file to a temporary directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall("/content/models_for_inference")
"""
    get_weights_from_git= """### Get Models and Weights"""
    code_6 = """
!wget https://raw.githubusercontent.com/YuvalRozner/WeatherNet/main/Backend/Model_Pytorch/utils/models_for_inference.zip
    """
    # Initialize initial_code_blocks with defined markdown and code
    initial_code_blocks = [
        (get_weights_from_git,code_6),
        (unzip_weights,code_5)
        #(markdown_1, code_1),
        #(markdown_1, code_2),
        #(markdown_1, code_3),

        # Add more tuples if needed
    ]

    # Hardcoded list of Python file paths
    python_files = [
        os.path.join(path_to_dir, "..", "common", "data.py"),                       # 0
        os.path.join(path_to_dir, "..", "common", "constantsParams.py"),            # 1    
        os.path.join(path_to_dir, "..", "common", "import_and_process_data.py"),    # 2    
        os.path.join(path_to_dir, "..", "AdvancedModel", "model.py"),               # 3
        os.path.join(path_to_dir, "..", "AdvancedModel", "parameters.py"),          # 4
        os.path.join(path_to_dir, "..", "AdvancedModel", "inference.py"),           # 5
    ]

    # Specify the output notebook path
    output_ipynb = os.path.join(path_to_dir, "inference.ipynb")

    # Define alteration rules
    # Example: Replace '/old/path/' with '/new/path/' in 'parameters.py'
    # and modify 'main.py' accordingly
    alteration_rules = {
        python_files[2] : {  # import_and_process_data.py
            "regex": [
                (r"from Model_Pytorch.common.constantsParams import \*", ""),
            ],
            "functions": [
                #"remove_code_above_main"
            ]
        },
        python_files[4]: { # parameters.py
            "regex": [
                (r"local_path", "colab_path"),
            ],
            "functions": [
                #"remove_code_above_main"
            ]
        },
        python_files[5]: {  # inference.py
            "regex": [
                (r"fileNames", "paths_in_colab"),
                (r"load_pkl_file", "pd.read_pickle"),
                (r"STATIONS_COORDINATES", "STATIONS_COORDINATES_COLAB"),
                (r"from Model_Pytorch.common.data import pd.read_pickle, normalize_coordinates", ""),
                (r"from Model_Pytorch.common.import_and_process_data import get_prccessed_latest_data_by_hour_and_station", ""),
                (r"from model import TargetedWeatherPredictionModel", ""),
                (r"from parameters import PARAMS, WINDOW_PARAMS, ADVANCED_MODEL_PARAMS, STATIONS_COORDINATES_COLAB, STATIONS_LIST, INFERENCE_PARAMS", ""),
                #(r"os\.path\.dirname\(__file__\)", "'/content/drive/MyDrive/final data'"),
            ],
            "functions": [
                #"remove_code_above_main"
            ]
        },
    }

    # Create the notebook with alterations and initial code blocks
    create_notebook_from_scripts(
        file_paths=python_files, 
        output_notebook=output_ipynb, 
        alteration_rules=alteration_rules,
        initial_code_blocks=initial_code_blocks  # Pass the initial code blocks here
    )
