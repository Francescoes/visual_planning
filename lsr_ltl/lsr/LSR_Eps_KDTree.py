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
#for kdtree
from sklearn.neighbors.kd_tree import KDTree
from sklearn.neighbors.ball_tree import BallTree
from sklearn.neighbors.dist_metrics import DistanceMetric


class LSR_Eps_KDTree(LSR):
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False, leaf_size = 40):

        super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.leaf_size = leaf_size
        self.tree = []

    #LSR phase 2 with kdtree*****************************************************
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
        H1 = self.G1.copy()
        Z_all = self.Z_all
        epsilon = self.epsilon
        distance_type = self.distance_type
        Z_sys_is=[]
        #metric
        if distance_type==np.inf:
            dist_metric = DistanceMetric.get_metric('chebyshev')
        else:
            dist_metric = DistanceMetric.get_metric('minkowski', p = distance_type)

        data_kdtree = np.array(self.Z_all_data)
        kdtree = KDTree(data_kdtree, metric=dist_metric, leaf_size = self.leaf_size)
        #kdtree = BallTree(data_kdtree, metric=dist_metric, leaf_size = self.leaf_size)

        #2.6 #while not Z_noachtion= null
        while len(Z_all) >0:
            if verbose:
                print("Z_all size: " + str(len(Z_all)))
            #2.1 randomly select z E Z_noaction
            z = random.choice(tuple(Z_all))
            #2.2 first time
            W_z=set()
            #z also belongs to the set
            W_z.add(z)
            #2.3 for all w E W_wz find W_w from 2.2 and set W_z:=W_z U W_w
            s_len_wz=len(W_z)
            #set init end length
            e_len_wz=np.Inf
            #check speedup
            W_w_to_check=W_z.copy()
            while not s_len_wz == e_len_wz:
                s_len_wz=len(W_z)
                #2.2 find all w E G1 for wich ||z-w||_d < 2*epsilon
                W_w=set()
                for w in W_w_to_check:
                    nearest_epsilon = kdtree.query_radius(data_kdtree[w:w+1], r=epsilon, return_distance = False)

                    #add indices
                    num_neigh = nearest_epsilon[0].size
                    for i in range(num_neigh):
                        curr_idx = int(nearest_epsilon[0][i])
                        #W_w.add(curr_idx)
                        if self.G1.nodes[curr_idx]['visited'] == 0:
                            W_w.add(curr_idx)
                            self.G1.nodes[curr_idx]['visited'] = 1

                W_w_to_check=W_w-W_w_to_check
                W_z=W_z.union(W_w)
                e_len_wz=len(W_z)
                #for speedup remove W_z from G1 copy
            #2.4 Z_noaction:=Z_noaztipn - W_z
            Z_all=Z_all - W_z
            #2.5 append W_z to a list
            Z_sys_is.append(W_z)

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
        self.tree = kdtree

    #Find closest nodes in graph G using distance type (1, 2, np.inf)
    def get_closest_nodes(self, z_pos_c1, z_pos_c2):
        keep_searching = True
        k = 1
        while keep_searching:
            (d_closest1, idx_g1_closest1) = self.tree.query(np.array([z_pos_c1]), k = k)
            idx_g1_closest1 = idx_g1_closest1[0][k-1]
            idx_lsr_closest1 = self.G1.nodes[idx_g1_closest1]['idx_lsr']
            if idx_lsr_closest1 >= 0:
                keep_searching = False
            k = k + 1


        keep_searching = True
        k = 1
        while keep_searching:
            (d_closest1, idx_g1_closest2) = self.tree.query(np.array([z_pos_c2]), k = k)
            idx_g1_closest2 = idx_g1_closest2[0][k-1]
            idx_lsr_closest2 = self.G1.nodes[idx_g1_closest2]['idx_lsr']
            if idx_lsr_closest2 >= 0:
                keep_searching = False
            k = k + 1

        return idx_lsr_closest1, idx_lsr_closest2
