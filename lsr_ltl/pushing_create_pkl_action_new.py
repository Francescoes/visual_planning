import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pickle
import os, os.path
import tkinter as tk
from tkinter import messagebox
import pickle
from unity_stacking_utils import get_class_from_filename, get_actions_from_classes

import torch
"""
Creates the pkl dataset given raw data into the format: [img1, img2, a, lambda, class1, class2]'
a = binary classification if action took place
lambda = detail about the action (dependedn on task)
"""




pkl_filename="./datasets/push_v1"


pkl_actions=[]

with open(pkl_filename + ".pkl", 'rb') as f:
    pkl_list = pickle.load(f)



random.seed(10)
# suffle the list
random.shuffle(pkl_list)


N = 5500
for k in range(0,N):
    img1 = pkl_list[k][0]
    img2 = pkl_list[k][1]
    action = pkl_list[k][2]
    A = pkl_list[k][3]
    pick_action = A[0:2]
    place_action = A[2:4]
    t_img1 = torch.tensor(img1/255.).float().permute(2, 0, 1)
    t_img2 = torch.tensor(img2/255.).float().permute(2, 0, 1)

    t_pick_action=torch.tensor(pick_action).float()
    t_place_action=torch.tensor(place_action).float()

    if (action==1):
        if  pick_action[0] == -1 or pick_action[1] == -1  or  place_action[0] == -1 or place_action[1] == -1 :
            print("Wait what!")
        else:
            pkl_actions.append((t_img1,t_img2,action,t_pick_action,t_place_action))





#Write pkl
with open("./action_data/push/push_actions.pkl", 'wb') as f:
    pickle.dump(pkl_actions, f)

print("saved: " + str(len(pkl_actions)))




pkl_actions_eval = []


N = 5500
N2 = len(pkl_list)
for k in range(N,N2):
    img1 = pkl_list[k][0]
    img2 = pkl_list[k][1]
    action = pkl_list[k][2]
    A = pkl_list[k][3]
    pick_action = A[0:2]
    place_action = A[2:4]
    t_img1 = torch.tensor(img1/255.).float().permute(2, 0, 1)
    t_img2 = torch.tensor(img2/255.).float().permute(2, 0, 1)

    t_pick_action=torch.tensor(pick_action).float()
    t_place_action=torch.tensor(place_action).float()

    if (action==1):
        if  pick_action[0] == -1 or pick_action[1] == -1  or  place_action[0] == -1 or place_action[1] == -1 :
            print("Wait what!")
        else:
            pkl_actions_eval.append((t_img1,t_img2,action,t_pick_action,t_place_action))




print("saved: " + str(len(pkl_actions_eval)))


# create also another pkl file for evaluation
with open("./action_data/push/evaluation_push_actions.pkl", 'wb') as f:
    pickle.dump(pkl_actions_eval, f)

#
