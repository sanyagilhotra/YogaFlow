import recom_custom_input
import pandas as pd
from recom_custom_input import set_of_joints
from recom_custom_input import get_selection
yog_benefits=pd.read_csv("Yog_poses.csv")
rec_poses=get_selection(yog_benefits=yog_benefits,joints_selected="Wrist Spine",level_of_difficulty='Intermediate')
print(rec_poses)