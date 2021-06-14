from __future__ import print_function
import argparse
import sys
from importlib.machinery import SourceFileLoader
import algorithms as alg
import torch
from torch.autograd import Variable
from dataloader import TripletTensorDataset
import architectures.VAE_ResNet as vae
import random
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
from numpy import linalg as LA
import statistics





pkl_filename="./datasets/push_v1_vae"

pkl_actions=[]

with open(pkl_filename + ".pkl", 'rb') as f:
    pkl_list = pickle.load(f)


random.seed(10)
# suffle the list
random.shuffle(pkl_list)

# Loas vae_b
distance_type=1

config_file="VAE_push_v1"
checkpoint_file="vae_lastCheckpoint.pth"




#load VAE
vae_config_file = os.path.join('.', 'configs', config_file + '.py')
vae_directory = os.path.join('.', 'models', checkpoint_file)
vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config
vae_config['exp_name'] = config_file
vae_config['vae_opt']['exp_dir'] = vae_directory
vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
vae_algorithm.model.eval()
print("loaded VAE")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cnt = 0
N = 6000
N= len(pkl_list)
dist = []
sum = 0
for k in range(0,N):

    action = pkl_list[k][2]

    # if action pair
    if (action==1):
        cnt = cnt +1
        img1 = pkl_list[k][0]/255.
        img2 = pkl_list[k][1]/255.
        A = pkl_list[k][3]
        pick_action = A[0:2]
        place_action = A[2:4]




        #get encoding
        img1=np.expand_dims(img1,axis=0)
        img2=np.expand_dims(img2,axis=0)
        #get recon start and goal and z
        x=torch.from_numpy(img1)
        x=x.float()
        x=x.permute(0,3,1,2)
        x = Variable(x).to(device)
        x2=torch.from_numpy(img2)
        x2=x2.float()
        x2=x2.permute(0,3,1,2)
        x2 = Variable(x2).to(device)
        dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(x)
        dec_mean2, dec_logvar2, z2, enc_logvar2=vae_algorithm.model.forward(x2)
        z_1=z[0].cpu().detach().numpy()
        z_2=z2[0].cpu().detach().numpy()


        # compute the d-norm of the difference
        n = LA.norm(z_1-z_2, distance_type)
        # n = torch.norm(z - z2, p=float(distance_type), dim=1).cpu().detach().numpy()
        sum = sum + n
        dist.append(n)


dm = sum/cnt
print("Action pairs")
print("cnt: ",cnt)
print("dm: ",dm)
print("min dist: ",min(dist))
print("max dist: ",max(dist))

print("median dist: ",statistics.median(dist))
#Write pkl
with open("./models/dm.pkl", 'wb') as f:
    pickle.dump(dm, f)




# For no-action pairs

cnt = 0
N = 5000

dist = []
sum = 0
for k in range(0,N):

    action = pkl_list[k][2]

    # if action pair
    if (action==0):
        cnt = cnt +1
        img1 = pkl_list[k][0]/255.
        img2 = pkl_list[k][1]/255.
        A = pkl_list[k][3]
        pick_action = A[0:2]
        place_action = A[2:4]




        #get encoding
        img1=np.expand_dims(img1,axis=0)
        img2=np.expand_dims(img2,axis=0)
        #get recon start and goal and z
        x=torch.from_numpy(img1)
        x=x.float()
        x=x.permute(0,3,1,2)
        x = Variable(x).to(device)
        x2=torch.from_numpy(img2)
        x2=x2.float()
        x2=x2.permute(0,3,1,2)
        x2 = Variable(x2).to(device)
        dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(x)
        dec_mean2, dec_logvar2, z2, enc_logvar2=vae_algorithm.model.forward(x2)
        z_1=z[0].cpu().detach().numpy()
        z_2=z2[0].cpu().detach().numpy()


        # compute the d-norm of the difference
        n = LA.norm(z_1-z_2, distance_type)
        # n = torch.norm(z - z2, p=float(distance_type), dim=1).cpu().detach().numpy()
        sum = sum + n
        dist.append(n)


dm = sum/cnt
print("No-Action pairs")
print("cnt: ",cnt)
print("dm: ",dm)
print("min dist: ",min(dist))
print("max dist: ",max(dist))

print("median dist: ",statistics.median(dist))
