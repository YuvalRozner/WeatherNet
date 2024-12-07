import pickle
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import threading
import os

# Global variables to control loading
loading_paused = False

def load_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file:\n{e}")
        return None

def load_and_display(file_path):
    data = load_pkl_file(file_path)
    if load_mode.get() == "all":
        display_data_all(data)
    else:
        display_data(data)

def display_data_incrementally(rows, columns, start=0, chunk_size=100):
    global loading_paused
    if loading_paused:
        root.after(100, display_data_incrementally, rows, columns, start, chunk_size)
        return

    # Insert a chunk of rows to avoid blocking the main thread
    end = start + chunk_size
    for row in rows[start:end]:
        tree.insert("", "end", values=row)
        progress_bar.step(1)
    if end < len(rows):
        root.after(100, display_data_incrementally, rows, columns, end, chunk_size)
    else:
        progress_bar.stop()

def display_data_all(data):
    if data is None:
        progress_bar.stop()
        return

    # Clear the treeview
    for item in tree.get_children():
        tree.delete(item)
    tree["columns"] = ()
    
    # Determine columns and prepare rows based on data type
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='w')
        rows = data.values.tolist()
        for row in rows:
            tree.insert("", "end", values=row)
    else:
        messagebox.showwarning("Warning", "Unsupported data format.")
        progress_bar.stop()

def display_data(data):
    if data is None:
        progress_bar.stop()
        return

    # Clear the treeview
    for item in tree.get_children():
        tree.delete(item)
    tree["columns"] = ()
    
    # Determine columns and prepare rows based on data type
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='w')
        
        total_rows = len(data)
        progress_bar['maximum'] = total_rows
        progress_bar['value'] = 0

        rows = data.values.tolist()
        display_data_incrementally(rows, columns)
    else:
        messagebox.showwarning("Warning", "Unsupported data format.")
        progress_bar.stop()

def open_file():
    global loading_paused
    loading_paused = False
    initial_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))  # Set to ../data folder
    file_path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        progress_bar['value'] = 0
        progress_bar['maximum'] = 100
        thread = threading.Thread(target=load_and_display, args=(file_path,))
        thread.start()

def pause_loading():
    global loading_paused
    loading_paused = not loading_paused
    pause_button.config(text="Resume" if loading_paused else "Pause")

#main funciton
if __name__ == "__main__":
    root = tk.Tk()
    root.title("PKL Viewer")

    # Set minimum and maximum window width size
    root.minsize(800, 600)
    root.maxsize(1200, 800)

    # Initialize load_mode variable after root window creation
    load_mode = tk.StringVar(value="chunks")

    # Create a frame for the radio buttons
    radio_frame = tk.Frame(root)
    radio_frame.pack()

    # Add radio buttons for load mode selection
    load_all_radio = tk.Radiobutton(radio_frame, text="Load All File", variable=load_mode, value="all")
    load_all_radio.pack(side=tk.LEFT)

    load_chunks_radio = tk.Radiobutton(radio_frame, text="Load in Chunks", variable=load_mode, value="chunks")
    load_chunks_radio.pack(side=tk.LEFT)

    # Create a Treeview widget to display the data
    tree = ttk.Treeview(root, show="headings")
    tree.pack(fill='both', expand=True)

    # Add a progress bar
    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate")
    progress_bar.pack(fill='x')

    open_button = tk.Button(root, text="Open PKL File", command=open_file)
    open_button.pack()

    pause_button = tk.Button(root, text="Pause", command=pause_loading)
    pause_button.pack()

    root.mainloop()