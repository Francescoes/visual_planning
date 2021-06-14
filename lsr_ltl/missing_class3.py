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

import matplotlib
matplotlib.use('TkAgg')


def main():

    # list of all possible classes
    all = []

    num1 = 3
    num2 = 3

    for i in range(9):
        free1 = list(range(9))
        free1.remove(i)
        for j in free1:
            free2 = list(range(9))
            free2.remove(i)
            free2.remove(j)
            for k in free2:
                class_img = np.zeros((num1*num2,1))

                class_img[i] = 1 # blue in position i
                class_img[j] = 2 # red in position j
                class_img[k] = 3 # gree in position k
                all.append(class_img.reshape((num1,num2)))


    print("len",len(all))

    # print(all)


    # print(np.zeros((num1*num2,1)).reshape((num1,num2)))


    # pkl file name
    pkl_filename = "./datasets/push_v13"
    pkl_list=[]

    with open(pkl_filename + ".pkl", 'rb') as f:
        pkl_list = pickle.load(f)


    random.seed(10)
    # suffle the list
    random.shuffle(pkl_list)



    N_train = 3500 # number of train samples


    list_train = pkl_list[:N_train]

    print("Size training dataset",len(list_train))



    classes = [] # list of classes
    for elem in list_train:
        classes.append(elem[4])
        classes.append(elem[5])

    print("number of classes ",len(classes))


    missing = all # list of classes not present in the dataset
    for elem in classes:

        for k,t in enumerate(missing):
            if (elem==t).all():
                missing.pop(k)




    print("Number of missing classes in the dataset ", len(missing))










if __name__== "__main__":
  main()
