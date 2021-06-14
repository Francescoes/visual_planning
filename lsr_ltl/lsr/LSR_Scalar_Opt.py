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
from lsr import LSR_Linkage,LSR_Eps_KDTree,LSR_Optics,LSR_MeanShift,LSR_HDBSCAN
#linkage hierachical clusters
from scipy.cluster.hierarchy import dendrogram, linkage, inconsistent, fcluster
import itertools
from scipy.optimize import minimize_scalar


class LSR_Scalar_Opt():
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m=0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False,
     opt_cost=("component", 10) ,lsr_method="linkage",lsr_linkage_method='single',lsr_optics_algorithm="kd_tree",lsr_meanshift_cluster_all=True,lower_b=0,upper_b=10):
        self.latent_map_file = latent_map_file
        self.epsilon = epsilon
        self.distance_type = distance_type
        self.graph_name = graph_name
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.min_edge_w = min_edge_w
        self.min_node_m = min_node_m
        self.directed_graph = directed_graph
        self.a_lambda_format = a_lambda_format
        self.verbose = verbose
        self.save_graph = save_graph

        #super().__init__(latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w, min_node_m, directed_graph,  a_lambda_format, verbose, save_graph)
        self.lsr_method = lsr_method
        self.opt_cost=opt_cost[0]
        self.c_max = opt_cost[1]
        self.lower_b = lower_b
        self.upper_b = upper_b
        self.lsr_linkage_method=lsr_linkage_method
        self.lsr_optics_algorithm=lsr_optics_algorithm
        self.lsr_meanshift_cluster_all=lsr_meanshift_cluster_all

        #initialize the seed
        np.random.seed(seed = 1977)

    def f_lsr(self,x0):
        print(x0)

        if self.lsr_method == "linkage":
            lsr_obj=LSR_Linkage.LSR_Linkage(self.latent_map_file, self.epsilon, self.distance_type, self.graph_name, self.config_file, self.checkpoint_file,
                self.min_edge_w, self.min_node_m, self.directed_graph,  self.a_lambda_format, self.verbose, self.save_graph
                , method=self.lsr_linkage_method,max_d=x0)

        if self.lsr_method == "eps":
            lsr_obj=LSR_Eps_KDTree.LSR_Eps_KDTree(self.latent_map_file, x0, self.distance_type, self.graph_name, self.config_file, self.checkpoint_file,
                self.min_edge_w, self.min_node_m, self.directed_graph,  self.a_lambda_format, self.verbose, self.save_graph, leaf_size = 40)

        if self.lsr_method == "optics":
            lsr_obj=LSR_Optics.LSR_Optics(self.latent_map_file, self.epsilon, self.distance_type, self.graph_name, self.config_file, self.checkpoint_file,
                self.min_edge_w, self.min_node_m, self.directed_graph,  self.a_lambda_format, self.verbose, self.save_graph
                , xi=x0, algorithm=self.lsr_optics_algorithm)

        if self.lsr_method == "meanshift":
            lsr_obj=LSR_MeanShift.LSR_MeanShift(self.latent_map_file, self.epsilon, self.distance_type, self.graph_name, self.config_file, self.checkpoint_file,
                self.min_edge_w, self.min_node_m, self.directed_graph,  self.a_lambda_format, self.verbose, self.save_graph
                , bandwidth=x0, cluster_all=self.lsr_meanshift_cluster_all)

        if self.lsr_method == "hdbscan":
            lsr_obj=LSR_HDBSCAN.LSR_HDBSCAN(self.latent_map_file, self.epsilon, self.distance_type, self.graph_name, self.config_file, self.checkpoint_file,
                self.min_edge_w, self.min_node_m, self.directed_graph,  self.a_lambda_format, self.verbose, self.save_graph
                , min_cluster_size=int(np.round(x0)))


        #we build the full stack
        G2, stats=lsr_obj.build_lsr(self.use_medoid, self.keep_biggest_component)
        self.G2 = G2
        self.stats = stats
        self.lsr_obj = lsr_obj

        #cost function:
        if self.opt_cost == "component":
            #calc opt criteria
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]

            max_num_nodes=-1
            num_components=len(S)
            for g in S:
                if g.number_of_nodes()>max_num_nodes:
                    max_num_nodes=g.number_of_nodes()
            #ceck for c_max
            print("num components: " + str(num_components) + " , nodes in biggest: " + str(max_num_nodes))
            if num_components>self.c_max:
                return np.inf
            else:
                return -max_num_nodes

        elif self.opt_cost == "component_all_nodes":
            #calc opt criteria
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]
            num_nodes = 0
            num_components=len(S)
            for g in S:
                num_nodes = num_nodes + g.number_of_nodes()
            if num_components>self.c_max:
                return np.inf
            else:
                return -num_nodes

        elif self.opt_cost == "component_all_edges":
            #calc opt criteria
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]
            num_edges = 0
            num_components=len(S)
            for g in S:
                num_edges = num_edges + g.number_of_edges()
            if num_components>self.c_max:
                return np.inf
            else:
                return -num_edges

        elif self.opt_cost == "component_all_edges_strong":
            #calc opt criteria
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.strongly_connected_components(G2)]
            num_edges = 0
            num_components=len(S)
            for g in S:
                num_edges = num_edges + g.number_of_edges()
            if num_components>self.c_max:
                return np.inf
            else:
                return -num_edges

        elif self.opt_cost == "component_connectivity":
            #calc opt criteria
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]
            second_eig = 0
            num_components=len(S)
            for g in S:
                h=g.to_undirected()
                spectrum = nx.laplacian_spectrum(h)
                spectrum = np.sort(spectrum)
                if h.number_of_nodes() >1:
                    second_eig = second_eig + spectrum[1]
            if num_components>self.c_max:
                return np.inf
            else:
                return -second_eig

        elif self.opt_cost == "action_pairs":
            scores = lsr_obj.score_clusters()
            a_score = scores[2]+scores[3]
            print("no action pairs: " + str(scores[2]) + " , action pairs: " + str(scores[3]))
            return -a_score

        elif self.opt_cost == "action_move":
            scores=lsr_obj.score_clusters()
            print("action 1 move error: " + str(scores[4]))
            return scores[4]

        elif self.opt_cost == "action_move_bounded":
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]
            num_components=len(S)
            print("num components: " + str(num_components) )
            if num_components>self.c_max:
                return np.inf
            else:
                scores=lsr_obj.score_clusters()
                print("action 1 move error: " + str(scores[4]))
                return scores[4]

        elif self.opt_cost == "action_move_factor_bounded":
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]
            num_components=len(S)
            print("num components: " + str(num_components) )
            if num_components>self.c_max:
                return np.inf
            else:
                scores=lsr_obj.score_clusters()
                info_score = scores[5]
                action_failures = info_score["1 move error on LSR"]+info_score["1 move error on LSR path to long"]
                print("action 1 move error: " + str(scores[4])+" fail count: "+str(info_score["1 move error on LSR"])+" too long: "+str(info_score["1 move error on LSR path to long"]))

                return scores[4]*(1+action_failures)

        elif self.opt_cost == "action_failures_bounded":
            if not self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
            if self.directed_graph:
                S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]
            num_components=len(S)
            print("num components: " + str(num_components) )
            if num_components>self.c_max:
                return np.inf
            else:
                scores=lsr_obj.score_clusters()
                info_score = scores[5]
                action_failures = info_score["1 move error on LSR"]+info_score["1 move error on LSR path to long"]
                print("fail count: "+str(info_score["1 move error on LSR"])+" too long: "+str(info_score["1 move error on LSR path to long"]))

                return action_failures

    def optimize_lsr(self, use_medoid = False, keep_biggest_component = False):
        self.use_medoid = use_medoid
        self.keep_biggest_component = keep_biggest_component
        res_opt = minimize_scalar(self.f_lsr, bounds=(self.lower_b ,self.upper_b ), method='bounded')
        return self.lsr_obj, res_opt, self.G2, self.stats

    # def lsr_phase_2(self):
    #     res = minimize_scalar(self.f_lsr, bounds=(self.lower_b ,self.upper_b ), method='bounded')
