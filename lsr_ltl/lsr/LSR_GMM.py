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
#for GMM
from sklearn.mixture import GaussianMixture


class LSR_GMM(LSR):
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False, n_components=300, covariance_type='full',init_params='kmeans'):

        super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init_params=init_params
        self.gmm = GaussianMixture(n_components=self.n_components,covariance_type=self.covariance_type,init_params=self.init_params)

    #LSR phase 2 GMM*****************************************************
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.score
    # INPUT:
    #   G1 - the first graph
    #   Z_all - all the encoded samples
    #   n_components - number of gaussian
    #   covariance_type - {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’} String describing the type of covariance parameters to use
    #   init_params - {‘kmeans’, ‘random’}
    #   verbose = False
    # OUTPUT:
    #   Z_sys_is - cluster with asigned nodes
    def lsr_phase_2(self):
        #Phase 2**********************************************************
        verbose = self.verbose
        Z_all = self.Z_all
        epsilon = self.epsilon
        Z_sys_is=[]

        Z_all_data = np.array(self.Z_all_data)

        #performe GMM
         #fit model
        print("fitting")
        self.gmm.fit(Z_all_data)
        #returns lable of cluster for each sample
        print("predicting")
        c_lables = self.gmm.predict(Z_all_data)

        #print(np.min(c_lables))
        #print(np.max(c_lables))

        num_c=max(c_lables)+1
        #prepare Z_sis_is
        Z_sys_is=[]
        for i in range(num_c):
        	W_z=set()
        	Z_sys_is.append(W_z)
        #add samples to the right set
        for g in self.G1:
        	#ignored samples are lables -1
                if not c_lables[g]==-1:
                    #print('idx: '+str(g)+', label: '+str(c_lables[g])+', lenZ: '+str(len(Z_sys_is)), ', maxclab: '+str(max(c_lables)))
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

    def get_LSR_node_pos(self, w_pos_all, W_z, g_idx):

        W_z_c_pos = np.mean(w_pos_all,axis=0)
        return W_z_c_pos
