import pickle 
import pandas as pd
import os

def load_pkl_file(file_name):
    current_path = os.path.dirname(__file__)
    file_path = f"{current_path}\\..\\..\\data\\{file_name}.pkl"
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"data succsesfuly loaded from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load file:\n{e}")
        return None
    

# main
if __name__ == "__main__":
    # Load the data
    data = load_pkl_file("Afeq")
    print(type(data))
