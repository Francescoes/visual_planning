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

    # pkl file name
    pkl_filename = "./datasets/push_v12"
    pkl_list=[]

    with open(pkl_filename + ".pkl", 'rb') as f:
        pkl_list = pickle.load(f)


    random.seed(10)
    # suffle the list
    random.shuffle(pkl_list)



    N_train = 1200 # number of train samples
    # N_train =  len(pkl_list)

    list = pkl_list[:N_train]

    print("Dataset size",len(list))



    classes = [] # list of classes
    for elem in list:
        classes.append(elem[4])
        classes.append(elem[5])




    print("number of classes ",len(classes))

    cl = [] # classes in the training dataset with no replicas
    for elem in classes:

        found = 0
        for t in cl:
            if (elem==t).all():
                found = 1
                break
        if not found:
            cl.append(elem)



    # class_temp = list[-1][5]
    # found = 0
    # for t in cl:
    #     if (class_temp==t).all():
    #         found = 1
    #         break
    # if not found:
    #     cl.append(class_temp)

    print("number of different classes present in the dataset",len(cl))


    ind=[] # list of number of times the corrispondent class is repeated
    for t in cl:
        cnt = 0
        for elem in classes:
            if (elem==t).all():
                cnt = cnt + 1

        ind.append(cnt)

    print("size ind",len(ind))
    print("max",max(ind))
    print("min",min(ind))
    print("The most repeated class \n",cl[ind.index(max(ind))])
    print("The least repeated class \n",cl[ind.index(min(ind))])

    h = [] # list of classes represented with the index of the corrisponding elem in cl
    for j,c in enumerate(classes):
        # print("j:",j)
        i = 0
        elem = cl[i]
        while not (c==elem).all():
            # print("\ti:",i)
            i +=1
            elem = cl[i]

        h.append(i)

    # print("Len of h", len(h))
    # for i in range(504):
    #     print(h.count(i))


    # i = 0
    # while classes:
    #     h.append(i)
    #     list(filter(lambda a: a != 2, x))
    #     cnt+=1



    plt.hist(h,bins=len(ind))

    plt.xlabel("class")
    plt.ylabel("number of replicas")
    plt.show()


    # import seaborn as sns
    # fig = plt.figure()
    # ax = sns.distplot(h,kde=False);
    # ax.set(xlabel='Class', ylabel='Frequency')
    #
    # plt.show()










if __name__== "__main__":
  main()
