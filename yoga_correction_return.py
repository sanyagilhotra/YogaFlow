# %%
#import packages
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
import matplotlib

from random import randint
from pathlib import Path
import json

from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet34
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# from custom_datasets import YogaPoseDataset

# import model_utils
# import plot_utils

# %%
frameWidth=-0
frameHeight=0
detected_keypoints=[]
keypoints_list=[]

# %%
#import model file and define pairs of pose
protoFile = "models/models/models/coco/pose_deploy_linevec.prototxt"
weightsFile = "models/models/models/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

# %%
# Find the Keypoints using Non Maximum Suppression on the Confidence Map
def getKeypoints(probMap, threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    
    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

# %%
# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB 
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)
                    
                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:            
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    # print(valid_pairs)
    return valid_pairs, invalid_pairs

# %%
# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score 
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

# %%
# #Loading Network and pass the image through networks
# t = time.time()
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# # Fix the input Height and get the width according to the Aspect Ratio
# inHeight = 368
# inWidth = int((inHeight/frameHeight)*frameWidth)

# inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
#                           (0, 0, 0), swapRB=False, crop=False)

# net.setInput(inpBlob)
# output = net.forward()
# print("Time Taken = {}".format(time.time() - t))

# %%
# #getting output of heatmap
# i = 0
# probMap = output[0, i, :, :]
# probMap = cv2.resize(probMap, (frameWidth, frameHeight))
# plt.figure(figsize=[14,10])
# plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# plt.imshow(probMap, alpha=0.6)
# plt.colorbar()
# plt.axis("off")

# %%
# #Detected keypoints
# detected_keypoints = []
# keypoints_list = np.zeros((0,3))
# keypoint_id = 0
# threshold = 0.1

# for part in range(nPoints):
#     probMap = output[0,part,:,:]
#     probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
# #     plt.figure()
# #     plt.imshow(255*np.uint8(probMap>threshold))
#     keypoints = getKeypoints(probMap, threshold)
#     # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
#     keypoints_with_id = []
#     for i in range(len(keypoints)):
#         keypoints_with_id.append(keypoints[i] + (keypoint_id,))
#         keypoints_list = np.vstack([keypoints_list, keypoints[i]])
#         keypoint_id += 1

#     detected_keypoints.append(keypoints_with_id)

# %%
# #Frame color
# frameClone = image1.copy()
# for i in range(nPoints):
#     for j in range(len(detected_keypoints[i])):
#         cv2.circle(frameClone, detected_keypoints[i][j][0:2], 3, [0,0,255], -1, cv2.LINE_AA)
# plt.figure(figsize=[15,15])
# plt.imshow(frameClone[:,:,[2,1,0]])

# %%
# #Finding valid pairs
# valid_pairs, invalid_pairs = getValidPairs(output)

# %%
# #Personwise keypoints
# personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

# %%
# #showing images
# for i in range(17):
#     for n in range(len(personwiseKeypoints)):
#         index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
#         if -1 in index:
#             continue
#         B = np.int32(keypoints_list[index.astype(int), 0])
#         A = np.int32(keypoints_list[index.astype(int), 1])
#         cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
        
# plt.figure(figsize=[15,15])
# plt.imshow(frameClone[:,:,[2,1,0]])

# %%
def display_keypoints(img_path):
    global frameWidth
    global frameHeight
    global detected_keypoints
    global keypoints_list
    image1 = cv2.imread(img_path)
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    #Loading Network and pass the image through networks
    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    #Detected keypoints
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
    #     plt.figure()
    #     plt.imshow(255*np.uint8(probMap>threshold))
        keypoints = getKeypoints(probMap, threshold)
        # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)
    frameClone = image1.copy()
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 3, [0,0,255], -1, cv2.LINE_AA)
    valid_pairs, invalid_pairs = getValidPairs(output)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
    #showing images
    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
    save_path="keypoint_path.png"    
    plt.figure(figsize=[15,15])
    plt.savefig(save_path)
    plt.imshow(frameClone[:,:,[2,1,0]])
    plt.savefig(save_path)
    return save_path
    

# %%
def extract_numeric_keypoints(img_path):
    global frameWidth
    global frameHeight
    global detected_keypoints
    global keypoints_list
    
    image1 = cv2.imread(img_path)
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    
    # Loading Network and pass the image through networks
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                     (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    
    # Detected keypoints
    detected_keypoints = []
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        if keypoints:  # Check if keypoints are detected
            detected_keypoints.extend(keypoints)
            if len(keypoints_list) == 0:
                keypoints_list = np.array(keypoints)
            else:
                keypoints_list = np.vstack([keypoints_list, keypoints])
            keypoint_id += len(keypoints)

    return [detected_keypoints]



def calculate_angles(keypoints):
    angles = []
    for kp_set in keypoints:
        # print("Keypoints set:", kp_set)
        for pair in POSE_PAIRS:
            # print("Pair:", pair)
            if len(kp_set) > max(pair):
                point_a = kp_set[pair[0]]
                point_b = kp_set[pair[1]]
                # print("Point A:", point_a)
                # print("Point B:", point_b)
                if point_a and point_b:
                    angle_rad = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)
    return angles

image_path = "test3.png"
detected_keypoints = extract_numeric_keypoints(image_path)
angles = calculate_angles(detected_keypoints)




# %%
# import csv
# import os

# # Function to append keypoints and angles to dataset CSV file
# def append_keypoints_and_angles_to_csv(dataset_csv_path, output_csv_path, image_folder):
#     with open(dataset_csv_path, 'r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         next(csv_reader)  # Skip header row
        
#         with open(output_csv_path, 'w', newline='') as output_csv_file:
#             csv_writer = csv.writer(output_csv_file)
#             csv_writer.writerow(['filename', 'pose_id', 'pose_name', 'keypoints', 'angles'])  # Write header
            
#             for i, row in enumerate(csv_reader, start=1):
#                 filename = row[0]  # Assuming filename is in the first column
#                 image_path = os.path.join(image_folder, filename)
                
#                 if not os.path.isfile(image_path):
#                     print(f"Image file not found: {image_path}")
#                     continue
                
#                 # Extract keypoints
#                 detected_keypoints = extract_numeric_keypoints(image_path)
                
#                 # Calculate angles
#                 angles = calculate_angles(detected_keypoints)
                
#                 # Append keypoints and angles to row
#                 row.append(detected_keypoints)
#                 row.append(angles)
                
#                 # Write updated row to output CSV file
#                 csv_writer.writerow(row)
                
#                 print(f"Processed image {i}: {filename}")



# dataset_csv_path = "new_relative_yoga_classifier.csv"
# output_csv_path = "angles.csv"
# image_folder = "resized_images"
# append_keypoints_and_angles_to_csv(dataset_csv_path, output_csv_path, image_folder)


# %%
import pandas as pd

df = pd.read_csv("angles.csv")

df = df[df['angles'].apply(lambda x: x != "[]")]

df.to_csv("angles.csv", index=False)



# %%
import os
import pandas as pd

# Load the dataset CSV file
dataset_df = pd.read_csv("angles.csv")

# Get the list of filenames from the CSV file
csv_filenames = dataset_df['filename'].tolist()

# Get the list of filenames from the dataset folder
dataset_folder_path = "resized_images"
dataset_filenames = [filename for filename in os.listdir(dataset_folder_path) if filename.endswith(".jpg")]

# Identify filenames not present in the CSV file
filenames_to_delete = [filename for filename in dataset_filenames if filename not in csv_filenames]

# Delete images corresponding to filenames not present in the CSV file
for filename in filenames_to_delete:
    file_path = os.path.join(dataset_folder_path, filename)
    os.remove(file_path)

# print("Deleted", len(filenames_to_delete), "images.")


# %%
import os

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
            else:
                print("Error loading image:", img_path)
    return images

# Example usage
dataset_folder_path = "resized_images"
dataset_images = load_images_from_folder(dataset_folder_path)
# print("Number of images loaded:", len(dataset_images))


# %%
import pandas as pd
import cv2
import numpy as np
import os

# Load dataset CSV file
dataset_df = pd.read_csv("angles.csv")

# Define keypoints mapping and pose pairs
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

def extract_numeric_keypoints(img_path):
    # Load image
    image1 = cv2.imread(img_path)
    if image1 is None:
        print("Error: Unable to load image at path:", img_path)
        return None
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    # Loading Network and pass the image through networks
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                     (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    # Detected keypoints
    detected_keypoints = []
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        probMap = cv2.threshold(probMap, threshold, 255, cv2.THRESH_BINARY)[1]
        keypoints = cv2.findNonZero(probMap)
        if keypoints is not None:
            keypoints_with_id = [(pt[0][0], pt[0][1], probMap[pt[0][1], pt[0][0]]) for pt in keypoints]
            detected_keypoints.append(keypoints_with_id)
        else:
            detected_keypoints.append([])
    
    return detected_keypoints

def calculate_angles(keypoints):
    angles = []
    for pair in POSE_PAIRS:
        point_a = keypoints[pair[0]] if len(keypoints) > pair[0] else None
        point_b = keypoints[pair[1]] if len(keypoints) > pair[1] else None
        if point_a and point_b:
            angle_rad = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
        else:
            angles.append(None)
    return angles

def calculate_angle_differences(keypoints1, keypoints2):
    angles_diff = []
    for i, (kp1, kp2) in enumerate(zip(keypoints1, keypoints2)):
        angle1 = calculate_angles(kp1)  
        angle2 = calculate_angles(kp2)  
        angle_diff = [abs(a1 - a2) if a1 is not None and a2 is not None else None for a1, a2 in zip(angle1, angle2)]
        angles_diff.append(angle_diff)
    return angles_diff



# %%
def deviation_check(img_path,pose_name):    
    # Load user's input pose image and extract keypoints
    user_pose_image_path = img_path;
    user_keypoints = extract_numeric_keypoints(user_pose_image_path)
    counter=0
    wrong_pose=0
    c=0
    pose_name = pose_name;  
    matching_poses = dataset_df[dataset_df['pose_name'] == pose_name]
    str1='Good Going!!'
    str2=''
    if len(matching_poses) == 0:
        print("No match found for the given pose name.")
    else:
        for idx, row in matching_poses.iterrows():
            if(counter>=1):
                break;
            dataset_pose_image_path = os.path.join("resized_images", row['filename'])
            dataset_keypoints = extract_numeric_keypoints(dataset_pose_image_path)  # Implement extract_numeric_keypoints function
            if dataset_keypoints:
                angle_differences = calculate_angle_differences(user_keypoints, dataset_keypoints)
                for i, diff in enumerate(angle_differences):
                    if(counter>=1):
                        break;
                    if any(angle_diff is not None and angle_diff > 10 for angle_diff in diff):
                        counter=counter+1;
                        str1="Angle deviation detected in pose: "+row['pose_name']+"\n"
                        for j, angle_diff in enumerate(diff):
                            if angle_diff is not None and angle_diff > 10 and angle_diff<45:  # Add this additional check
                                if (j>=len(keypointsMapping)):
                                    break;
                                c=c+1;
                                str2=str2+"Joint: "+ str(keypointsMapping[j])+" deviates by "+str(angle_diff)+" degrees."+"\n"
                            if(angle_diff>=100):
                                wrong_pose=wrong_pose+1
                    else:
                        str1="Pose matched with "+row['pose_name']+"\n"
                        str2="Good going!!"
                        counter=counter+1;
                if(c!=0):
                    return str1+str2;
            else:
                return("Error: Unable to extract keypoints from the dataset pose image at path")
        return "Pose matched with "+row['pose_name']+"\n"+"Good going!!"


# %%



