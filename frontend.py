import pandas as pd
from recom_custom_input import set_of_joints
from recom_custom_input import get_selection
import tkinter as tk
from tkinter import ttk
from tkinter import Tk, Checkbutton, IntVar, Frame, Label, Button,END,ACTIVE
from functools import partial
from tkinter import filedialog
from PIL import Image, ImageTk
from yog_correction_script import display_keypoints
from tkinter import Listbox, SINGLE
from yoga_correction_return import deviation_check
yog_benefits = pd.read_csv("Yog_poses.csv")
# rec_poses = get_selection(yog_benefits=yog_benefits, joints_selected="Wrist Spine Arms", level_of_difficulty='Advanced')
# print(rec_poses)
selected_tasks = []  # Empty list to store selected joints
rec_label = None
results_of_comparison=''
name_of_pose=''
rec_poses=[]
def choose(index, task):
    # Update selected_joints based on checkbox state
    if var_list[index].get() == 1:
        selected_tasks.append(task)
    else:
        if task in selected_tasks:
            selected_tasks.remove(task)

def imageUploader():
    global results_of_comparison
    fileTypes = [("Image files", "*.jpg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)
    
    if len(path):
        img = Image.open(path)
        img = img.resize((200, 200))
        display_keypoints(path)
        results_of_comparison=deviation_check(path,name_of_pose)
        pic = ImageTk.PhotoImage(img)
        keypoint_path="keypoint_path.png"
        k_img=Image.open(keypoint_path)
        k_img=k_img.resize((250,250))
        k_pic=ImageTk.PhotoImage(k_img)
    
        label_img.config(image=pic)
        label_img.image = pic
        key_img.config(image=k_pic)
        key_img.image=k_pic
        label_correction.config(text=results_of_comparison)
    # if no file is selected, then we are displaying below message
    else:
        print("No file is Choosen !! Please choose a file.")



selected_label = None 
rec_label= None 
difficulty_select_label=None
selected_primary_joint = None

pose_radiobuttons = []  # Global list to store radio buttons
pose_radiobutton_var = None  # Global variable for radio button variable

def update_selected_pose():
    global name_of_pose
    selected_pose_index = pose_radiobutton_var.get()
    if selected_pose_index != -1:
        name_of_pose = rec_poses[selected_pose_index]

def yog_print(selected_text, difficulty_text):
    global rec_label
    global pose_radiobuttons
    global pose_radiobutton_var
    global rec_poses
    # get primary joint recommendation
    rec_primary_pose=get_selection(yog_benefits=yog_benefits, joints_selected=selected_primary_joint, level_of_difficulty=difficulty_text,num_of_poses=30)
    # Get recommended yoga poses based on selected joints and difficulty level
    rec_poses = get_selection(yog_benefits=rec_primary_pose, joints_selected=selected_text, level_of_difficulty=difficulty_text,num_of_poses=10)
    
    # Destroy previously created radio buttons
    if pose_radiobuttons:
        for rb in pose_radiobuttons:
            rb.destroy()
    
    # Create a single IntVar for all radio buttons
    pose_radiobutton_var = IntVar()
    # Create radio buttons for each recommended yoga pose
    for pose in rec_poses:
        rb = ttk.Radiobutton(root, text=pose, variable=pose_radiobutton_var, value=rec_poses.index(pose))
        rb.grid(column=0, row=6 + rec_poses.index(pose), padx=10, pady=5)
        pose_radiobuttons.append(rb)
    # Update the global list of radio buttons
    pose_radiobuttons = pose_radiobuttons
    pose_radiobutton_var.trace_add('write', lambda *args: update_selected_pose())





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
    global selected_primary_joint
    selected_text=''
    # Clear the selection in the primary joints listbox
    primary_joints_listbox.selection_clear(0, END)

    # Retrieve the selected primary joint name
    selected_primary_joint_index = primary_joints_listbox.index(ACTIVE)
    if selected_primary_joint_index != -1:
        selected_primary_joint = primary_joints_listbox.get(selected_primary_joint_index)
        selected_text=selected_primary_joint+" "
    else:
        selected_primary_joint = None
    selected_text =selected_text+ ' '.join(selected_tasks)
    if selected_label is None:
        selected_label = Label(root, text="Selected Joints:")
        selected_label.grid(column=0, row=3, padx=10, pady=5)
    selected_label.config(text="Selected Joints: "+selected_text)
    selected_tasks.clear()
    for var in var_list:
        var.set(0)
    difficulty_label_text = get_difficulty_level_label()
    if difficulty_select_label is None:
        difficulty_select_label = Label(root, text="Selected Difficulty Level:")
        difficulty_select_label.grid(column=0, row=4, padx=10, pady=5)
    difficulty_select_label.config(text="Selected Level: "+difficulty_label_text)
    yog_print(selected_text,difficulty_label_text)





root = Tk()
root.title("YogaFlow")
Label(root, text='Joints',font="TkDefaultFont 15 bold").grid(row=0,column=0)
root.minsize(510,800)
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
        checkbutton.grid(row=row, column=col, padx=14, pady=5)  # Add padding
        
        col += 1
        if col == 4:
            col = 0
            row += 1

difficulty_label = ttk.Label(frame, text="Difficulty Level:",font="TkDefaultFont 15 bold")
difficulty_label.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")

# Create a new frame for the radio buttons
difficulty_frame = Frame(frame)
difficulty_frame.grid(row=4, column=1, columnspan=3, padx=13, pady=5, sticky="nsew")

# Create IntVar variable for difficulty level
difficulty_var = IntVar()

# Create Radiobutton widgets for difficulty levels
basic_radiobutton = ttk.Radiobutton(difficulty_frame, text="Basic", variable=difficulty_var, value=1)
intermediate_radiobutton = ttk.Radiobutton(difficulty_frame, text="Intermediate", variable=difficulty_var, value=2)
hard_radiobutton = ttk.Radiobutton(difficulty_frame, text="Hard", variable=difficulty_var, value=3)

# Grid layout for difficulty level radio buttons
basic_radiobutton.grid(row=0, column=0, padx=12, pady=5)
intermediate_radiobutton.grid(row=0, column=1, padx=12, pady=5)
hard_radiobutton.grid(row=0, column=2, padx=12, pady=5)

primary_joints=set_of_joints

stop_button = Button(root, text="Stop Selection", command=stop_selection,padx=13)
stop_button.grid(column=0, row=2) 

uploadButton = tk.Button(root, text="Locate Image",command=imageUploader,padx=13)
uploadButton.grid(column=0, row=16) 
label_img = tk.Label(root)
key_img=tk.Label(root)
label_img.grid(column=0, row=17, sticky="w",padx=(10,2)) 
key_img.grid(column=0, row=17,sticky='w',padx=(250,2))
primary_joints_label = Label(frame, text="Primary Joint:", font="TkDefaultFont 15 bold")
primary_joints_label.grid(row=0, column=5, padx=20, pady=5, sticky="nsew")
label_correction=tk.Label(root)
label_correction.grid(column=0,row=17,sticky='w',padx=(500,2))

# Create Listbox for primary joints
primary_joints_listbox = Listbox(frame, selectmode=SINGLE)
for joint in primary_joints:
    primary_joints_listbox.insert(END, joint)
primary_joints_listbox.grid(row=0, column=6, rowspan=5, padx=14, pady=5, sticky="nsew")

# Function to handle selection from the primary joints listbox
def select_primary_joint():
    selected_index = primary_joints_listbox.curselection()
    if selected_index:
        selected_joint = primary_joints_listbox.get(selected_index[0])
        choose(selected_index[0], selected_joint)

# Bind a function to handle selection events
primary_joints_listbox.bind("<<ListboxSelect>>", lambda event: select_primary_joint())

root.mainloop()