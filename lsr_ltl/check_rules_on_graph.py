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
    config_file = "VAE_push_v1"
    dataset_name="push_v1"
    graph_name=config_file  +"_graph"

    image_save_name="./imgs/image"



    #load graph
    G = nx.read_gpickle('graphs/'+graph_name+'.pkl')





    LIST = []
    LIST_OF_NODES = list(G.nodes)

    wrong = []


    # dictionary with first arg the node index and second arg the list of neighbors
    LIST_OF_NEIGHBORS  = {l: list(G.neighbors(l)) for l in LIST_OF_NODES}

    # print(LIST_OF_NEIGHBORS.get(552))
    # for l in LIST_OF_NODES:
    #     LIST_OF_NEIGHBORS.append(list(G.neighbors(l)))

    print ("number_of_nodes: ", len(LIST_OF_NODES))
    # print ("number_of_neig: ", len(LIST_OF_NEIGHBORS))

    # print(LIST_OF_NEIGHBORS)
    # lista = LIST_OF_NEIGHBORS[447]
    # print("List of neighbors",lista)
    # print("List of neighbors",list(G.neighbors(447)))

    # print(LIST_OF_NODES)

    # for l in LIST_OF_NODES:
    #     print(l,LIST_OF_NODES.index(l)-l)

    while len(LIST_OF_NODES)>0:
        i = random.choice(LIST_OF_NODES)

        # print("Selected node i ",i)
        if not LIST_OF_NEIGHBORS[i]: # if the node does not have neighbors
            LIST_OF_NODES.remove(i)
            print("removed node ",i)
        else:
            # print("List of neighbors",LIST_OF_NEIGHBORS.get(i))
            class_i = G.nodes[i]['_class']
            for j in LIST_OF_NEIGHBORS.get(i):
                # print("For ",i,",",j)
                class_j = G.nodes[j]['_class']
                if  is_okay(class_i,class_j):
                    # remove the edge if it is everything okay
                    LIST_OF_NEIGHBORS[i].remove(j)

                else:
                    print("Found a pair of nodes connected in a wrong way: node ",i," and node ",j)
                    # print(class_i)
                    # print(class_j)
                    # exit(0)
                    wrong.append([i,j])

                    LIST_OF_NEIGHBORS[i].remove(j)
                    break
                #     exit()



    # lista = LIST_OF_NEIGHBORS.get(0)
    # print("List of neighbors",lista)
    # for j in lista:
    #             print(j)


    if wrong:
        print("Something wrong")
        print("Number of forbidden transitions ",len(wrong))
    else:
        print("It seems that all the transitions are okay!")
    #
    # print("List of duplicated states:")
    # for l in LIST:
    #     if len(l)>1:
    #         print(l)

            # z_list = []
            # if save_img:
            #     for i in l:
            #         z_pos=G.nodes[i]['pos']
            #         z_list.append(z_pos)
            #         z_pos = torch.from_numpy(z_pos).float().to(device)
            #         z_pos = z_pos.unsqueeze(0)
            #
            #         img_rec,_=vae_algorithm.model.decoder(z_pos)
            #
            #         img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
            #
            #         #make to 255
            #         t_img=img_rec_cv*255
            #         t_img_f=t_img.astype("uint8").copy()
            #
            #
            #         cv2.imwrite(image_save_name+str(i)+".png",t_img_f)
            #
            #     print(np.linalg.norm(z_list[0]-z_list[1],ord=distance_type))




if __name__== "__main__":
  main()
