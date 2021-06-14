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
from lsr import LSR_Linkage
#linkage hierachical clusters
from scipy.cluster.hierarchy import dendrogram, linkage, inconsistent, fcluster
import itertools


class LSR_Linkage_bsearch(LSR):
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False,
     method='single',bmin_d=0,bmax_d=300,bdeapth=10,c_max=10):

        super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.method = method
        self.bmin_d = bmin_d
        self.bmax_d = bmax_d
        self.bdeapth = bdeapth
        self.c_max = c_max

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
        verbose = self.verbose
        Z_all = self.Z_all
        epsilon = self.epsilon
        distance_type = self.distance_type
        Z_sys_is=[]

        Z_all_data = np.array(self.Z_all_data)
        #format distance types
        if distance_type==1:
        	metric='cityblock'
        if distance_type==2:
        	metric='euclidean'
        if distance_type==np.inf:
        	metric='chebyshev'

        #calculate dendogram (Z)
        Z = linkage(Z_all_data, method = self.method,metric = metric)



        #do intervall search to get a good max_d
        # for bivarient search we do need a notion of which direction is better (+ - or faluir cases) I think
        # So here I check intervals and narrow down a good max_d assuming there is a single hill between min and max
        # We sould check if there are better ways to do this :(
        max_d=(self.bmax_d-self.bmin_d)/2
        max_d_l=self.bmin_d
        max_d_h=self.bmax_d
        for i in range(self.bdeapth):
            #self.lsr_phase_1(self.latent_map_file, self.directed_graph)
            print("-----")
            print("max_d "+ str(max_d))
            max_d_l_p=max_d_l
            max_d_h_p=max_d_h
            #do 2 searches left and right of the current value and see which one is better
            max_d_h=max_d+(max_d_h-max_d)/2

            c_lables= fcluster(Z, max_d_h, criterion='distance')
            num_c=len(set(c_lables))
            print(num_c)
            Z_sys_is=[]
            for i in range(num_c):
                W_z=set()
                Z_sys_is.append(W_z)
            for g in self.G1:
                idx_zsys = c_lables[g]-1
                Z_sys_is[idx_zsys].add(g)
                self.G1.nodes[g]['idx_lsr'] = idx_zsys


            self.Z_sys_is = Z_sys_is

            



            #we call the normal Linakge with the max_d_h
            lsr_obj=LSR_Linkage.LSR_Linkage(self.latent_map_file, self.epsilon, self.distance_type, self.graph_name, self.config_file, self.checkpoint_file, 
                self.min_edge_w, self.min_node_m, self.directed_graph,  self.a_lambda_format, self.verbose, self.save_graph 
                , method=self.method,max_d=max_d_h)
            #we build the full stack
            G2,stats=lsr_obj.build_lsr() 
            #self.G2=G2  

            S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            max_num_nodes_h=-1
            num_components_h=0
            for g in S:
                num_components_h+=1
                if g.number_of_nodes()>max_num_nodes_h:
                    max_num_nodes_h=g.number_of_nodes()


            #calculate the scores:
            scores=self.score_clusters()

            w_z_min=np.Inf
            w_z_max=-np.Inf
            for W_z in Z_sys_is:
                if len(W_z)<w_z_min:
                    w_z_min=len(W_z)
                if len(W_z) > w_z_max:
                    w_z_max=len(W_z)

            

            a_score_h=scores[2]+scores[3]
            n_cluster_h=len(self.Z_sys_is)
            #n_cluster_h=w_z_max
            #n_cluster_h=max_num_nodes
            print(str(max_d_h) + "  : scoresh: " + str(scores[2])+" , " + str(scores[3])+ " , " + str(max_num_nodes_h))
            print(scores)



            # correct_pairs = [0, 0]
            # incorrect_pairs = [0, 0]
            # for W_z in Z_sys_is:
            #     W_z_labels=[]
            #     if len(W_z)>0:
            #         for w in W_z:
                       
            #             current_node = self.G1.nodes[w]

            #             #get pair spec
            #             pair_spec = current_node['pair_spec']
            #             lsr_node = current_node['idx_lsr']

            #             if lsr_node >= 0 and len(pair_spec) > 0:
            #                 action = pair_spec[0]
            #                 pair_node_idx = pair_spec[1]

            #                 #get the node of the second state
            #                 pair_node = self.G1.nodes[pair_node_idx]
            #                 pair_lsr_node = pair_node['idx_lsr']
            #                 if pair_lsr_node < 0:
            #                     #the node ofthe second state has been pruned
            #                     count_removed_nodes[action] = count_removed_nodes[action] + 1
            #                     #print("pruned second")
            #                 else:
            #                     if action == 0 and pair_lsr_node == lsr_node:
            #                         correct_pairs[action] = correct_pairs[action] + 1
            #                     elif action == 0 and pair_lsr_node != lsr_node:
            #                         incorrect_pairs[action] = incorrect_pairs[action] + 1
            #                     elif action == 1 and pair_lsr_node != lsr_node:
            #                         correct_pairs[action] = correct_pairs[action] + 1
            #                     elif action == 1 and pair_lsr_node == lsr_node:
            #                         incorrect_pairs[action] = incorrect_pairs[action] + 1

            #             elif lsr_node < 0 and len(pair_spec) > 0:
            #                 #the node has been pruned
            #                 action = pair_spec[0]
            #                 count_removed_nodes[action] = count_removed_nodes[action] + 1


            # correct_noac = correct_pairs[0]/self.no_action_count
            # correct_ac = correct_pairs[1]/self.action_count
            # print("ooooooooooooooooooooooo")
            # print(correct_noac)
            # print(correct_ac)
            # print("ooooooooooooooooooooooo")

            # #do a action check:
            # if correct_ac==1:
            #     f = open(self.latent_map_file, 'rb')
            #     latent_map = pickle.load(f)
            #     for latent_pair in latent_map[:1000]:
            #         z_pos_c1=latent_pair[0]
            #         z_pos_c2=latent_pair[1]
            #         action=latent_pair[2]
            #         # find the node:
            #         if action==1:
            #             for W_z in Z_sys_is:
            #                 for w in W_z:

            #                     if (self.G1.nodes[w]['pos']==z_pos_c1).all():
            #                         for w2 in W_z:
            #                             if not w ==w2:
            #                                 if (self.G1.nodes[w2]['pos']==z_pos_c2).all():
            #                                     a=1/0

            #     for W_z in Z_sys_is:
            #         for w in W_z:
            #             current_node = self.G1.nodes[w]
            #             pair_spec = current_node['pair_spec']
            #             lsr_node = current_node['idx_lsr']
            #             #not an empty one
            #             if len(pair_spec)>1:
            #                 pair_node_idx = pair_spec[1]
            #                 pair_lsr_node=self.G1.nodes[pair_node_idx]['idx_lsr']
            #                 if pair_spec[0]==1:
            #                     if pair_lsr_node == lsr_node:
            #                         print("here")
            #                         print()
            #                         print(pair_lsr_node)
            #                         print(lsr_node)
            #                         print()
            #                         a=1/0
            #                     for w2 in W_z:
            #                         if not w == w2:                                        
            #                             if self.G1.nodes[w2]['idx_lsr']!= self.G1.nodes[w]['idx_lsr']:
            #                                 print()
            #                                 print(self.G1.nodes[lsr_node]['idx_lsr'])
            #                                 print(self.G1.nodes[pair_node_idx]['idx_lsr'])
            #                                 print()

            #                                 a=1/0

            


            # now for lower
            max_d_l=max_d-(max_d-max_d_l)/2
           
            # c_lables= fcluster(Z, max_d_l, criterion='distance')
            # num_c=len(set(c_lables))
            # Z_sys_is=[]
            # for i in range(num_c):
            #     W_z=set()
            #     Z_sys_is.append(W_z)
            # for g in self.G1:
            #     idx_zsys = c_lables[g]-1
            #     Z_sys_is[idx_zsys].add(g)
            #     self.G1.nodes[g]['idx_lsr'] = idx_zsys

            # self.Z_sys_is = Z_sys_is


             #we call the normal Linakge with the max_d_l
            lsr_obj=LSR_Linkage.LSR_Linkage(self.latent_map_file, self.epsilon, self.distance_type, self.graph_name, self.config_file, self.checkpoint_file, 
                self.min_edge_w, self.min_node_m, self.directed_graph,  self.a_lambda_format, self.verbose, self.save_graph 
                , method=self.method,max_d=max_d_l)
            #we build the full stack
            G2,stats=lsr_obj.build_lsr()  
            #self.G2=G2

            S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            max_num_nodes_l=-1
            num_components_l=0

            for g in S:
                num_components_l+=1
                if g.number_of_nodes()>max_num_nodes_l:
                    max_num_nodes_l=g.number_of_nodes()


            #calculate the scores:
            scores=self.score_clusters()
            w_z_min=np.Inf
            w_z_max=-np.Inf
            for W_z in Z_sys_is:
                if len(W_z)<w_z_min:
                    w_z_min=len(W_z)
                if len(W_z) > w_z_max:
                    w_z_max=len(W_z)

            a_score_l=scores[2]+scores[3]
            n_cluster_l=len(self.Z_sys_is)
            #n_cluster_l=w_z_max
            #n_cluster_l=max_num_nodes

            print(str(max_d_l) + " scoresl: " + str(scores[2])+" , " + str(scores[3])+ " , " + str(max_num_nodes_l))
            print(scores)

            

            # if np.round(a_score_l,2)>np.round(a_score_h,2):
            #     max_d=max_d_l
            #     max_d_l=max_d_l_p
            # elif np.round(a_score_l,2)<np.round(a_score_h,2):
            #     max_d=max_d_h
            #     max_d_h=max_d_h_p
            # else:
            #     if n_cluster_l<n_cluster_h:
            #         max_d=max_d_l
            #         max_d_l=max_d_l_p
            #     elif n_cluster_l>n_cluster_h:
            #         max_d=max_d_h
            #         max_d_h=max_d_h_p
            #     else:
            #         print("same")
            #         if scores[2]+scores[3] >=1.8:
            #             break
            #         else:
            #             #random restart?
            #             print("restarting")
            #             max_d=random.randint(self.bmin_d,self.bmax_d)
            #             max_d_l=self.bmin_d
            #             max_d_h=self.bmax_d

            #optimise first for c_max
            if num_components_h>self.c_max or num_components_l> self.c_max:
                #check wher it is at least better
                if num_components_h>num_components_l:
                    max_d=max_d_l
                    max_d_l=max_d_l_p
                elif num_components_h>num_components_l:
                    max_d=max_d_h
                    max_d_h=max_d_h_p
                else:
                    if bool(random.getrandbits(1)):
                        max_d=max_d_h
                        max_d_h=max_d_h_p
                    else:
                        max_d=max_d_l
                        max_d_l=max_d_l_p
            else:               

                if max_num_nodes_h<max_num_nodes_l:
                    max_d=max_d_l
                    max_d_l=max_d_l_p
                elif max_num_nodes_h>max_num_nodes_l:
                    max_d=max_d_h
                    max_d_h=max_d_h_p
                else:
                    print("equal")
                    break


                #need something for this part ...
                #max_d=random.randint(self.bmin_d,self.bmax_d)

            

        # #final path with the midle value of the corners found
        # max_d=(max_d_h-max_d_l)/2
        
        # #calculate dendogram (Z)
        # Z = linkage(Z_all_data, method = self.method,metric = metric)

        # # build cluster using max distance in the dendogram to find it ... this should be automated TODO!
        # c_lables= fcluster(Z, max_d, criterion='distance')
        # #find number of cluster
        # num_c=len(set(c_lables))
        # #prepare Z_sis_is
        # Z_sys_is=[]
        # for i in range(num_c):
        #     W_z=set()
        #     Z_sys_is.append(W_z)
        # #add samples to the right set
        # #cluster indexing starts at 1
        # for g in self.G1:
        #     # at the idx i, there will be all the samples with cluster i
        #     Z_sys_is[c_lables[g]-1].add(g)

        #Print result of phase 2
        Z_sys_is=self.Z_sys_is
        if True:
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
