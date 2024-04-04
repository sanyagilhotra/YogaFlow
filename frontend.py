import pandas as pd
from recom_custom_input import set_of_joints
from recom_custom_input import get_selection
import tkinter as tk
from tkinter import ttk
from tkinter import Tk, Checkbutton, IntVar, Frame, Label, Button
from functools import partial

yog_benefits = pd.read_csv("Yog_poses.csv")
rec_poses = get_selection(yog_benefits=yog_benefits, joints_selected="Wrist Spine Arms", level_of_difficulty='Advanced')
# print(rec_poses)
selected_tasks = []  # Empty list to store selected tasks
rec_label = None
selected_text=''
def choose(index, task):
    # Update selected_tasks based on checkbox state
    if var_list[index].get() == 1:
        selected_tasks.append(task)
    else:
        if task in selected_tasks:
            selected_tasks.remove(task)


selected_label = None 
rec_label= None 
difficulty_select_label=None
def yog_print(selected_text,difficulty_text):
    global rec_label
    rec_poses = get_selection(yog_benefits=yog_benefits, joints_selected=selected_text, level_of_difficulty=difficulty_text)
    yog_text = '\n'.join(rec_poses)
    if rec_label is None:
        rec_label = Label(root, text="")
        rec_label.grid(column=0, row=5, padx=10, pady=10)
    
    rec_label.config(text="Yoga Poses: \n"+yog_text)

def get_difficulty_level_label():
    if difficulty_var.get() == 1:
        return "Basic"
    elif difficulty_var.get() == 2:
        return "Intermediate"
    elif difficulty_var.get() == 3:
        return "Hard"
    else:
        return "Intermediate"  # No difficulty level selected
    
def stop_selection():
    global selected_label
    global selected_text
    global difficulty_select_label
    selected_text=''
    selected_text = ' '.join(selected_tasks)
    if selected_label is None:
        selected_label = Label(root, text="Selected Joints:")
        selected_label.grid(column=0, row=3, padx=10, pady=10)
    selected_label.config(text="Selected Joints: "+selected_text)
    selected_tasks.clear()
    for var in var_list:
        var.set(0)
    difficulty_label_text = get_difficulty_level_label()
    if difficulty_select_label is None:
        difficulty_select_label = Label(root, text="Selected Difficulty Level:")
        difficulty_select_label.grid(column=0, row=4, padx=10, pady=10)
    difficulty_select_label.config(text="Selected Level: "+difficulty_label_text)
    yog_print(selected_text,difficulty_label_text)
root = Tk()

Label(root, text='Joints',font="TkDefaultFont 15 bold").grid(column=0, row=0)
root.minsize(450,800)
# Create a 4x4 grid frame
frame = Frame(root)
frame.grid(column=0, row=1)

# Initialize variables and create checkboxes in a grid
var_list = []

row = 0
col = 0

for index, task in enumerate(set_of_joints):
    if row < 4 and col < 4:  # Limit to 4 rows and 4 columns
        var_list.append(IntVar(value=0))
        checkbutton = Checkbutton(frame, variable=var_list[index], text=task, command=partial(choose, index, task))
        checkbutton.grid(row=row, column=col, padx=5, pady=5)  # Add padding
        
        col += 1
        if col == 4:
            col = 0
            row += 1
difficulty_label = ttk.Label(frame, text="Difficulty Level:",font="TkDefaultFont 15 bold")
difficulty_label.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

# Create a new frame for the radio buttons
difficulty_frame = Frame(frame)
difficulty_frame.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky="nsew")

# Create IntVar variable for difficulty level
difficulty_var = IntVar()

# Create Radiobutton widgets for difficulty levels
basic_radiobutton = ttk.Radiobutton(difficulty_frame, text="Basic", variable=difficulty_var, value=1)
intermediate_radiobutton = ttk.Radiobutton(difficulty_frame, text="Intermediate", variable=difficulty_var, value=2)
hard_radiobutton = ttk.Radiobutton(difficulty_frame, text="Hard", variable=difficulty_var, value=3)

# Grid layout for difficulty level radio buttons
basic_radiobutton.grid(row=0, column=0, padx=5, pady=5)
intermediate_radiobutton.grid(row=0, column=1, padx=5, pady=5)
hard_radiobutton.grid(row=0, column=2, padx=5, pady=5)

stop_button = Button(root, text="Stop Selection", command=stop_selection)
stop_button.grid(column=0, row=2) 


root.mainloop()
