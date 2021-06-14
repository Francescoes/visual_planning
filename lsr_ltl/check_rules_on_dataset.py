# functions to produce figures

from __future__ import print_function
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
import algorithms as alg
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from dataloader import TripletTensorDataset
import architectures.VAE_ResNet as vae
import cv2
import pickle
import networkx as nx
import random
import lsr_utils as lsr

# 0 the box did not move
# 1 the box moved according to rules
# -1 the box moved not according to rules
def did_it_move_correctly(idx_1,idy_1,idx_2,idy_2):
    if (idx_1-idx_2==0) and (idy_1-idy_2==0):
        return 0
    elif (not(idx_1-idx_1==0)) and (not(idy_1-idy_2==0)):
         return -1 # both x and y changed
    else:
        return 1




def is_okay(class1,class2):

    cnt = 0 # number of box which moved

    idx1_1,idy1_1 = np.where(class1==1)
    idx1_2,idy1_2 = np.where(class2==1)

    if list(idx1_1): # if it is not empty
        check = did_it_move_correctly(idx1_1,idy1_1,idx1_2,idy1_2)
        if check==-1:# if the box moved not according the rules
            print("box 1 moved not according the rules")
            return 0
        elif check ==1:
            cnt = cnt +1



    idx2_1,idy2_1 = np.where(class1==2)
    idx2_2,idy2_2 = np.where(class2==2)

    if list(idx2_1): # if it is not empty
        check = did_it_move_correctly(idx2_1,idy2_1,idx2_2,idy2_2)
        if check==-1:# if the box moved not according the rules
            print("box 2 moved not according the rules")
            return 0
        elif check ==1:
            cnt = cnt +1


    idx3_1,idy3_1 = np.where(class1==3)
    idx3_2,idy3_2 = np.where(class2==3)

    if list(idx3_1): # if it is not empty
        check = did_it_move_correctly(idx3_1,idy3_1,idx3_2,idy3_2)
        if check==-1:# if the box moved not according the rules
            print("box 3 moved not according the rules")
            return 0
        elif check ==1:
            cnt = cnt +1


    if cnt >1:
        print("more than 1 box moved")
        return 0
    elif cnt ==0:
        print("nothing changed")
        return 0
    else:
        print("the transition is okay")
        return 1



def main():

    pkl_filename="./datasets/push_v1"


    pkl_actions=[]

    with open(pkl_filename + ".pkl", 'rb') as f:
        pkl_list = pickle.load(f)


    wrong = []


    N = len(pkl_list)
    for k in range(0,N):
        class_i = pkl_list[k][4]
        class_j = pkl_list[k][5]
        action = pkl_list[k][2]


        if (action==1):
            if  not is_okay(class_i,class_j):
                print(class_i,"\n",class_j)
                exit(0)
                wrong.append(k)





    if wrong:
        print("Something wrong")
        print(wrong)
    else:
        print("It seems that all the transitions are okay!")






if __name__== "__main__":
  main()
