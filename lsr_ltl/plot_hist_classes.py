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
    distance_type=1
    weight=1.0
    config_file="VAE_push_v1"
    checkpoint_file="vae_lastCheckpoint.pth"
    output_file="labeled_latent_spaces/"+config_file+"_latent_space_map"
    dataset_name="push_v1"
    graph_name=config_file  +"_graph"

    image_save_name="./imgs/image"







    #load graph
    G = nx.read_gpickle('graphs/'+graph_name+'.pkl')





    print ("number_of_nodes: ", G.number_of_nodes())

    LIST = []
    LIST_OF_NODES = list(G.nodes)






    cnt = [] # list of the number of times the class occurs
    index=[]
    # print(LIST_OF_NODES)
    for idx,l in enumerate(LIST_OF_NODES):
        n = 1
        # print("\n",l)
        class_l = G.nodes[l]['_class']
        index.append(idx)
        for k in LIST_OF_NODES:
            if not(k==l):
                class_k = G.nodes[k]['_class']
                if ((class_l==class_k).all()):
                    n = n+1
                    LIST_OF_NODES.remove(k)
                    index.append(idx)


        cnt.append(n)




    cl = []
    for l in LIST_OF_NODES:
        _class = G.nodes[l]['_class']

        found = 0
        for t in cl:
            if (_class==t).all():
                found = 1
                break
        if not found:
            cl.append(_class)






    print("number of different classes",len(cl))
    print("check that I have the number of replicas for each class",len(cnt))
    print("sum of the number of replicas for each class",sum(cnt))
    print("check indexes",len(index))

    # print(index)
    #
    # print(cnt)
    plt.hist(index, bins=len(cl))
    plt.xlabel("class")
    plt.ylabel("number of replicas")
    plt.show()

    with open("./check_eq/classes.pkl", 'wb') as f:
        pickle.dump(cl, f)

if __name__== "__main__":
  main()
