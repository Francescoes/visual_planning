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
#for meanshift
from sklearn.cluster import MeanShift


class LSR_MeanShift(LSR):
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False, bandwidth=None, cluster_all=True):

        super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.bandwidth = bandwidth
        self.cluster_all = cluster_all
        self.clustering = []

    #LSR phase 2 with mean shift*****************************************************
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    # INPUT:
    #   G1 - the first graph
    #   Z_all - all the encoded samples
    #   epsilon - epsilon parameter
    #   verbose = False
    # OUTPUT:
    #   Z_sys_is - cluster with assigned nodes
    def lsr_phase_2(self):
        #Phase 2**********************************************************
        verbose = self.verbose
        Z_all = self.Z_all
        epsilon = self.epsilon
        distance_type = self.distance_type
        Z_sys_is=[]

        Z_all_data = np.array(self.Z_all_data)

        #perform meanshift
        #returns lable of cluster for each sample
        #c_lables = MeanShift(bandwidth=self.bandwidth, cluster_all=self.cluster_all).fit_predict(Z_all_data)
        #find number of cluster
        # num_c=len(set(c_lables))
        # for i in range(num_c):
        # 	W_z = set()
        # 	Z_sys_is.append(W_z)
        # #add samples to the right set
        # for g in self.G1:
        # 	Z_sys_is[c_lables[g]].add(g)

        clustering = MeanShift(bandwidth = self.bandwidth, cluster_all = self.cluster_all).fit(Z_all_data)

        #find number of cluster
        num_c = len(clustering.cluster_centers_)
        print('Num clusters')
        print(num_c)
        for i in range(num_c):
        	W_z = set()
        	Z_sys_is.append(W_z)
        # unique_cluster = set(clustering.labels_)
        # print(unique_cluster)

        #add samples to the right set
        for g in self.G1:
            # at the idx i, there will be all the samples with cluster i
        	Z_sys_is[clustering.labels_[g]].add(g)


        for i in range(num_c):
            if len(Z_sys_is[i]) ==0:
                print(i)




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
        self.clustering = clustering


    def get_LSR_node_pos(self, w_pos_all, W_z, g_idx):
        return self.get_LSR_node_pos_default(w_pos_all, W_z, g_idx)



    # #Find closest nodes in graph G using distance type (1, 2, np.inf)
    # def get_closest_nodes(self, z_pos_c1, z_pos_c2):
    #
    #
    #     idx_cluster_closest1 = self.clustering.predict(np.array([z_pos_c1]))
    #     idx_cluster_closest2 = self.clustering.predict(np.array([z_pos_c2]))
    #
    #     # the cluster index coincides with the lsr graph index
    #     idx_lsr_closest1 = idx_cluster_closest1[0]
    #     idx_lsr_closest2 = idx_cluster_closest2[0]
    #
    #
    #     return idx_lsr_closest1, idx_lsr_closest2
