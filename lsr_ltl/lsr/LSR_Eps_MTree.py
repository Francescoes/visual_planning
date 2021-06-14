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
#mtree
from mtree import MTree
import itertools
from mtree import functions



#has a dic for M-tree
class hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self))


class LSR_Eps_MTree(LSR):
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False, min_node_capacity = 50):

        super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.min_node_capacity = min_node_capacity
        self.tree = []


    #LSR phase 2 with mtree*****************************************************
    # It generates  Z_sys_is - cluster with assigned nodes
    def lsr_phase_2(self):
        #Phase 2**********************************************************
        verbose = self.verbose
        H1 = self.G1.copy()
        G1 = self.G1
        Z_all = self.Z_all
        epsilon = self.epsilon
        distance_type = self.distance_type
        min_node_capacity = self.min_node_capacity
        Z_sys_is=[]
        #build m-tree
        if distance_type==1:
            mtree = MTree(min_node_capacity=min_node_capacity,distance_function = functions.L1_distance)
        if distance_type ==2:
            mtree = MTree(min_node_capacity=min_node_capacity,distance_function = functions.L2_distance)
        if distance_type == np.inf:
            mtree = MTree(min_node_capacity=min_node_capacity,distance_function = functions.Linf_distance)

        for g in G1:
            #make node hashible
            h_tree_obj=hashabledict(G1.nodes[g])
            mtree.add(h_tree_obj)

        # #2.6 #while not Z_noaction= null
        # iter = 1
        # while iter<2: #len(Z_all) >0:
        #     iter = iter +1
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

                    w_pos=G1.nodes[w]['pos']
                    nearest_epsilon=list(mtree.get_nearest(G1.nodes[w], range=epsilon))

                    #find tree idx
                    mtree_idx=[]
                    for i in range(len(nearest_epsilon)):
                        W_w.add(int(nearest_epsilon[i][0]['idx']))
                        #remove from tree
                        mtree.remove(hashabledict(G1.nodes[int(nearest_epsilon[i][0]['idx'])]))

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
        self.tree = mtree


    # #Find closest nodes in graph G using distance type (1, 2, np.inf)
    # def get_closest_nodes(self, z_pos_c1, z_pos_c2):
    #     # Create a temporary node
    #     node_to_copy = self.G1.nodes()[0]
    #     tmp_node = node_to_copy.copy()
    #     tmp_node['idx'] = -1
    #
    #     # node_g1_closest1 = list(self.tree.get_nearest( self.G1.nodes()[0], limit = 1))
    #     # print(node_g1_closest1)
    #     # print(self.tree.root.data)
    #
    #     tmp_node['pos'] = z_pos_c1
    #     node_g1_closest1 = list(self.tree.get_nearest( tmp_node, limit = 1))
    #
    #
    #     tmp_node['pos'] = z_pos_c2
    #     node_g1_closest2 = list(self.tree.get_nearest( tmp_node, limit = 1))
    #     print(node_g1_closest1)
    #     print(node_g1_closest2)
    #     print(self.tree.root.data)
    #     node_g1_closest1 = node_g1_closest1[0][0]
    #     node_g1_closest2 = node_g1_closest2[0][0]
    #
    #     idx_lsr_closest1 = node_g1_closest1['idx_lsr']
    #     idx_lsr_closest2 = node_g1_closest2['idx_lsr']
    #
    #     return idx_lsr_closest1, idx_lsr_closest2
