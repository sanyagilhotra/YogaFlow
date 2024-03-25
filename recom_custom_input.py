# %%
import pandas as pd
import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
yog_benefits=pd.read_csv("Yog_poses.csv")


# %%
yog_benefits.head()

# %%
# type(yog_benefits['Joints Targeted New'][0])

# %%
str=''
for i in range(0,81):
    str=str+yog_benefits['Joints Targeted New'][i]+" "

# %%
list_of_joints=str.split()

# %%
list_of_joints

# %%
set_of_joints=set(list_of_joints)

# %%
set_of_joints

# %%
class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, pose, recom_pose,counter,dif_level):
        rec_items = len(recom_pose)
        recommended_poses=[]
        # print(f'Yoga Poses:')
        j=0
        for i in range(0,80):
            if dif_level=='Intermediate':
                if(j<counter and (recom_pose[i][3]=='Basic' or recom_pose[i][3]=='Intermediate')):
                    # print(f"Pose {j+1}:",end=' ')
                    # print(f"{recom_pose[i][2]}") 
                    # print("--------------------")
                    recommended_poses.append(recom_pose[i][2])
                    j=j+1
            elif dif_level=='Basic':
                if(j<counter and (recom_pose[i][3]=='Basic')):
                    # print(f"Pose {j+1}:",end=' ')
                    # print(f"{recom_pose[i][2]}") 
                    # print("--------------------")
                    recommended_poses.append(recom_pose[i][2])
                    j=j+1
            else:
                if(j<counter and (recom_pose[i][3]=='Basic' or recom_pose[i][3]=='Intermediate' or recom_pose[i][3]=='Advanced')):
                    # print(f"Pose {j+1}:",end=' ')
                    # print(f"{recom_pose[i][2]}") 
                    # print("--------------------")
                    recommended_poses.append(recom_pose[i][2])
                    j=j+1
        return recommended_poses
            
        
    def recommend(self, recommendation):
        # Get joints to find recommendations for
        joint_poses = recommendation['poses']
        # Get number of poses to recommend
        number_poses = recommendation['number_of_poses']
        # Get the difficulty of the poses
        difficulty=recommendation['difficulty']
        # Get the number of most similar similars from matrix similarities
        recom_pose = self.matrix_similar[joint_poses]
        # print each item
        rec_poses=self._print_message(pose=joint_poses, recom_pose=recom_pose,counter=number_poses,dif_level=difficulty)
        return rec_poses

# %%
def get_selection(yog_benefits,joints_selected,level_of_difficulty):
    yog_benefits=yog_benefits
    x={'Yoga Poses':"User 1",'Joints Targeted New':joints_selected}
    x_df = pd.DataFrame([x])
    yog_benefits = pd.concat([yog_benefits, x_df], ignore_index=True)
    joints=yog_benefits['Joints Targeted New']
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    pose_matrix = tfidf.fit_transform(joints)
    cosine_similarities = cosine_similarity(pose_matrix) 
    similarities = {}
    for i in range(len(cosine_similarities)):
        similar_indices = cosine_similarities[i].argsort()[:-50:-1] 
        similarities[yog_benefits['Joints Targeted New'].iloc[i]] = [(cosine_similarities[i][x], yog_benefits['Joints Targeted New'].iloc[x], yog_benefits['Yoga Poses'][x],yog_benefits['Levels of Difficulty'].iloc[x]) for x in similar_indices][1:]
    recommedations = ContentBasedRecommender(similarities)
    recommendation2 = {
    "poses": yog_benefits['Joints Targeted New'].iloc[81],
    "number_of_poses": 10,
    "difficulty":level_of_difficulty
    }
    rec_poses=recommedations.recommend(recommendation2)
    return rec_poses

# %%
joints=yog_benefits['Joints Targeted New']
# print(type(joints))

# %%
joints

# %%
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
lyrics_matrix = tfidf.fit_transform(joints)


# %%
cosine_similarities = cosine_similarity(lyrics_matrix) 
similarities = {}

# %%
# get_selection(yog_benefits=yog_benefits,joints_selected="Hips Lower_Back Wrists",level_of_difficulty='Basic')

# %%
# import tkinter as tk
# win=tk.Tk()
# win.title("YogaFlow")
# win.minsize(600,600)
# win.mainloop()


