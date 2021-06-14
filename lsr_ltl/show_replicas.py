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





def main():
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
    LIST_OF_NODES = list(G.nodes) # list of nodes of the graph
    # print(LIST_OF_NODES)
    for l in LIST_OF_NODES:

        print("\n",l)
        class_l = G.nodes[l]['_class']
        list_l = [] # list of replicas of node l
        list_l.append(l)
        aux = LIST_OF_NODES.copy()
        for k in aux:
            if not(k==l):
                class_k = G.nodes[k]['_class']
                if ((class_l==class_k).all()):
                    # print(LIST_OF_NODES)
                    list_l.append(k)
                    LIST_OF_NODES.remove(k)
                    # print("k==l - deleted k =",k)


        if len(list_l)>1:
            LIST.append(list_l)



    print("Lenght of list after pruning the replicas ",len(LIST_OF_NODES))
    print("Number of nodes with at least a replica ",len(LIST))

    print("List of duplicated states:")
    # for l in LIST:
    #     if len(l)>1:
    #         print(l)



if __name__== "__main__":
  main()
