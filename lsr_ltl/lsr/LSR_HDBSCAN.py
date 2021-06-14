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
#for HDBSCAN
import hdbscan


class LSR_HDBSCAN(LSR):
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False, a_lambda_format='stacking', verbose = False, save_graph = False, min_cluster_size=2):

        super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.min_cluster_size = min_cluster_size

    #LSR phase 2 HDBSCAN*****************************************************
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#r2c55e37003fe-1
    # INPUT:
    #   G1 - the first graph
    #   Z_all - all the encoded samples
    #   min_cluster_size - must be greater than 1 , outlier and noise removal (we do this after the clustering not during)
    #   verbose = False
    # OUTPUT:
    #   Z_sys_is - cluster with asigned nodes
    def lsr_phase_2(self):
        #Phase 2**********************************************************
        verbose = self.verbose
        Z_all = self.Z_all
        epsilon = self.epsilon
        distance_type = self.distance_type
        Z_sys_is=[]

        Z_all_data = np.array(self.Z_all_data)

        #performe OPTICS
        #returns lable of cluster for each sample
        if distance_type==np.inf:
            dist_metric = 'infinity'
        elif distance_type == 1:
            dist_metric = 'l1'
        elif distance_type == 2:
            dist_metric = 'l2'


        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric=dist_metric)
        c_lables  = clusterer.fit_predict(Z_all_data)

        #print(np.min(c_lables))
        #print(np.max(c_lables))
        #a=1/0

        #find number of cluster
        if np.min(c_lables)==-1:
        	num_c=len(set(c_lables))-1
        else:
        	num_c=len(set(c_lables))

        #prepare Z_sis_is
        Z_sys_is=[]
        for i in range(num_c):
        	W_z=set()
        	Z_sys_is.append(W_z)
        #add samples to the right set
        for g in self.G1:
        	#ignored samples are lables -1
        	if not c_lables[g]==-1:
        		Z_sys_is[c_lables[g]].add(g)

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
        print(len(Z_sys_is))

    def get_LSR_node_pos(self, w_pos_all, W_z, g_idx):
        return self.get_LSR_node_pos_default(w_pos_all, W_z, g_idx)
