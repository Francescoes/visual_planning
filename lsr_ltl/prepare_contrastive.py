import pickle
import math
import random
import numpy as np
import random
import sys
import torch


import cv2

import argparse
import os
from importlib.machinery import SourceFileLoader
import algorithms as alg
from torch.autograd import Variable
from dataloader import TripletTensorDataset
import architectures.VAE_ResNet as vae
import networkx as nx
import lsr_utils as lsr

from numpy import linalg as LA







with open("./datasets/push_v12.pkl", 'rb') as f:
    all2 = pickle.load(f)


with open("./datasets/push_v13.pkl", 'rb') as f:
    all3 = pickle.load(f)


all23 = all2+all3


random.seed(0)
random.shuffle(all23)


pkl_list_new = []





# Contrastive pairs in dataset 2 bxes´ configs
save_img = 0
cnt=0 # number of no action pairs counted as action pairs
cnt2=0 # number of contrastive pairs between 2 and 3 boxes configs
k=0
while k<2000:
    print("#",k)

    i = random.randint(0,len(all23)-1)
    tuple1 = all23[i]
    class_1 = tuple1[4]

    j = random.randint(0,len(all23)-1)
    tuple2 = all23[j]

    print(i)
    print(j)
    class_2 = tuple2[4]

    if (not i ==j):
        if ((class_1==class_2).all()):
            cnt += 1
        if not (np.count_nonzero(class_1) == np.count_nonzero(class_2)):
            cnt2 += 1



        pkl_list_new.append((tuple1[0], tuple2[0], 1, [-1,-1,1,-1,-1], 1, 1))
        k += 1

        if save_img:
            cv2.imwrite("./Img_contrastive/"+ str(i) + "_"+ str(j) +"_0.png",tuple1[4])
            cv2.imwrite("./Img_contrastive/"+ str(i) + "_"+ str(j) +"_1.png",tuple2[4])



print("Size of contrastive pairs dataset: ", len(pkl_list_new))
print("Percentage of no-action pairs counted as action pairs: ", cnt/len(pkl_list_new)*100,"%")
print("Percentage of actually contrastive pairs (2-3 boxes pairs): ", cnt2/len(pkl_list_new)*100,"%")




with open("./datasets/push_v12.pkl", 'rb') as f:
    push_v12 = pickle.load(f)
with open("./datasets/push_v13.pkl", 'rb') as f:
    push_v13 = pickle.load(f)

pkl_list = push_v12+push_v13
random.seed(10)
random.shuffle(pkl_list)


list = pkl_list[:5000] + pkl_list_new



print("Size of push_v1_vae: ", len(list))



with open("./datasets/push_v1_vae.pkl", 'wb') as f:
    pickle.dump(list, f)
