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
    - str: The code with lines above 'if __name__ == "__main__":' removed.
    """
    lines = code.splitlines()
    new_lines = []
    main_found = False
    
    for idx, line in enumerate(lines):
        if 'if __name__ == "__main__":' in line:
            main_found = True
            # Optionally include the 'if __name__ == "__main__":' line
            # Uncomment the next line if you want to keep it
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

def create_notebook_from_scripts(file_paths, output_notebook, alteration_rules=None):
    """
    Creates a Jupyter Notebook with each Python file's content in separate code cells,
    applying any specified alterations.
    
    Parameters:
    - file_paths (list of str): List of paths to the Python (.py) files.
    - output_notebook (str): Path where the output .ipynb file will be saved.
    - alteration_rules (dict): Optional. A dictionary mapping file paths to their
      respective alteration rules, which can include regex patterns and functions.
    """
    # Create a new notebook object
    nb = new_notebook()

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
            markdown_cell = f"### `{path}`"
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




    # Hardcoded list of Python file paths
    python_files = [
        os.path.join(path_to_dir, "..", "common", "data.py"),                                   # 0   
        os.path.join(path_to_dir, "..", "common", "window_generator_multiple_stations.py"),      # 1  
        os.path.join(path_to_dir, "..", "AdvancedModel", "model.py"),      # 2
        os.path.join(path_to_dir, "..", "AdvancedModel", "train.py"),      # 3
        os.path.join(path_to_dir, "..", "AdvancedModel", "parameters.py"), # 4
        os.path.join(path_to_dir, "..", "AdvancedModel", "main.py"),       # 5
    ]

    # Specify the output notebook path
    output_ipynb = os.path.join(path_to_dir, "AdvancedModel.ipynb")

    # Define alteration rules
    # Example: Replace '/old/path/' with '/new/path/' in 'script2.py'
    # and replace 'DEBUG = True' with 'DEBUG = False' in 'script3.py'
    alteration_rules = {
        python_files[4]: {
            "regex": [
                (r"# parametes.py", "# parametes.ipynb"),
                (r"# parametes.py", "# parametes.ipynb"),
            ],
            "functions": [
                
            ]
        },
        python_files[5]: {
            "regex": [
                (r"fileNames", "paths_in_colab"),
                (r"load_pkl_file", "pd.read_pickle"),
                (r"STATIONS_COORDINATES", "STATIONS_COORDINATES_COLAB"),
                (r"os\.path\.dirname\(__file__\)", "'out'"),
            ],
            "functions": [
                "remove_code_above_main"

            ]
        },
    }

    # Create the notebook with alterations
    create_notebook_from_scripts(python_files, output_ipynb, alteration_rules)
