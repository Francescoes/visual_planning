# Implementation of Latent space roadmap functions for the paper :
# Latent Space Roadmap for Visual Action Planning of Deformable and Rigid Object Manipulation
#-----------------------------------------------------------------------------------------------

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
import time
from lsr.LSR import LSR
#linkage hierachical clusters
from minisom import MiniSom
import itertools


class LSR_SOM_SIMPLE(LSR):
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False, method='single',max_d=100,som_max=10000):

        super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.method = method
        self.max_d = max_d
        self.som_max=som_max

    #LSR phase 2 Linkage (Hirachical clustering) *****************************************************
    # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.linkage.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.inconsistent.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    # INPUT:
    #   G1 - the first graph
    #   Z_all - all the encoded samples
    #   distance_type - L1, L2, or Linf distance
    #   cluster_all - If true, then all points are clustered, even those orphans that are not within any kernel
    #	method - typy of linkage : single, complete, average, weighted, centroid, median, ward
    #	incon_depth - depth to calculate inconsitancy from
    #   verbose = False
    # OUTPUT:
    #   Z_sys_is - cluster with asigned nodes
    def lsr_phase_2(self):
        #Phase 2**********************************************************
        distance_type = self.distance_type
        verbose = self.verbose
        if distance_type==1:
            metric='manhattan'
        if distance_type==2:
            metric='euclidean'
        if distance_type==np.inf:
            metric='chebyshev'

        som_data=np.array(self.Z_all_data)

        #build a sefl organising map and use the neuronse that assigne the itmes as "clusters"
        #reduce the size of the map iterativly by substrating the "empty" neurons and rebuilding it
        som_size=int(np.sqrt(self.som_max))
        feature_size=som_data.shape[1]
        activation_maps=[]
        while True:
            print("som size: " + str(som_size) + "x" + str(som_size) + "= " + str(som_size*som_size))
            som = MiniSom(som_size,som_size, feature_size, sigma=4,
              learning_rate=0.5, neighborhood_function='triangle',activation_distance=metric)

            #train som
            som.pca_weights_init(som_data)
            som.train(som_data, 15000, random_order=True, verbose=True)  # random training

            #check all the data and detect how many neurons are used/active
            #this is very crude as we always use a square .... :(
            wmap = np.zeros((som_size,som_size),dtype=int)
            for x in som_data:
                w = som.winner(x)
                wmap[w[0]][w[1]]+=1

            activation_maps.append(wmap)

            

            #check how many 0 ther are:
            unused_count=0
            for i in range(wmap.shape[0]):
                for j in range(wmap.shape[1]):
                    if wmap[i][j]==0:
                        unused_count+=1

            print("found " + str(unused_count) + " inactive neurons of " + str(som_size**2))

            if unused_count==0:
                break
            #calc new som size:
            som_size=int((np.sqrt(som_size**2-unused_count)))


        #debug activation maps:
        # print(len(activation_maps))
        # for wmap in activation_maps:
        #     plt.figure()
        #     plt.title('z as 2d heat map')
        #     p = plt.imshow(wmap)
        #     plt.colorbar(p)
        #     plt.show()

        # a=1/0

        #now we build the clusters
        som = MiniSom(som_size,som_size, feature_size, sigma=4,
              learning_rate=0.5, neighborhood_function='triangle',activation_distance=metric)

        #train som
        som.pca_weights_init(som_data)
        som.train(som_data, 15000, random_order=True, verbose=True)  # random training

        wmap = [set() for _ in range(som_size*som_size)]
        # for i in range(som_size):
        #     for j in range(som_size):
        #         k=i*som_size+j
        #         wmap[k]=set()

        for x in som_data:
            w = som.winner(x)
            #this is a terrible lookup :(
            for g in self.G1:
                if (x==self.G1.nodes[g]['pos']).all():
                    k=w[0]*som_size+w[1]
                    wmap[k].add(g)

        #now transfer it to old format ...
        Z_sys_is=wmap


        
        # distance_type = self.distance_type
        # Z_sys_is=[]

        # Z_all_data = np.array(self.Z_all_data)
        # #format distance types
        

        # #calculate dendogram (Z)
        # Z = linkage(Z_all_data, method = self.method,metric = metric)



        # # build cluster using max distance in the dendogram to find it ... this should be automated TODO!
        # c_lables= fcluster(Z, self.max_d, criterion='distance')

        # #find number of cluster
        # num_c=len(set(c_lables))
        # #prepare Z_sis_is
        # Z_sys_is=[]
        # for i in range(num_c):
        # 	W_z=set()
        # 	Z_sys_is.append(W_z)
        # #add samples to the right set
        # #cluster indexing starts at 1
        # for g in self.G1:
        #     # at the idx i, there will be all the samples with cluster i
        # 	Z_sys_is[c_lables[g]-1].add(g)

        #Print result of phase 2
        if verbose:
            print("***********Phase two done*******")
            print("Num disjoint sets: " + str(len(Z_sys_is)))
            num_z_sys_nodes=0
            w_z_min=np.Inf
            w_z_max=-np.Inf
            for W_z in Z_sys_is:
                if len(W_z)<w_z_min:
                    w_z_min=len(W_z)
                if len(W_z) > w_z_max:
                    w_z_max=len(W_z)
                num_z_sys_nodes+=len(W_z)
            print("Total number of components: " + str(num_z_sys_nodes))
            print("Max number W_z: " + str(w_z_max)+ " min number w_z: " + str(w_z_min))

        self.Z_sys_is = Z_sys_is


    def get_LSR_node_pos(self, w_pos_all, W_z, g_idx):

        return self.get_LSR_node_pos_default(w_pos_all, W_z, g_idx)
