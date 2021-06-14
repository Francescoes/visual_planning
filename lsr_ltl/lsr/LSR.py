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
import math
from sklearn.cluster import DBSCAN




#Format distance type from string to correct format
def format_distance_type(distance_type):
    if distance_type=='inf' or distance_type==np.inf:
        return np.inf
    else:
        return int(distance_type)


class LSR:
    def __init__(self, latent_map_file, epsilon, distance_type, graph_name, config_file, checkpoint_file, min_edge_w=0, min_node_m = 0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False, save_graph = False):
        self.latent_map_file = latent_map_file
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.graph_name = graph_name

        self.directed_graph = directed_graph
        self.verbose = verbose
        self.save_graph = save_graph
        self.a_lambda_format = a_lambda_format

        self.epsilon = epsilon
        self.min_edge_w = min_edge_w
        self.min_node_m = min_node_m

        self.distance_type = format_distance_type(distance_type)
        #Variables for lsr
        self.Z_all=[]
        self.Z_all_data=[]
        self.Z_sys_is = []
        self.G1 = []
        self.G2 = []
        self.stats_dict = []
        self.h_s=-1
        self.h_c=-1
        self.eps_approximation_computed = False

    def build_lsr(self, use_medoid = False, keep_biggest_component = False):

        self.use_medoid = use_medoid
        self.keep_biggest_component = keep_biggest_component
        # Build initial graph with all the nodes and edges
        self.lsr_phase_1(self.latent_map_file, self.directed_graph)
        #Clustering phase
        self.lsr_phase_2()
        #Build graph
        self.lsr_phase_3(self.config_file, self.checkpoint_file, self.a_lambda_format)
        self.lsr_phase_4(self.graph_name, self.min_edge_w , self.min_node_m, self.save_graph)
        return self.G2, self.stats_dict



    def compute_medoid(self, in_set):
        distance_type = self.distance_type
        n_el = len(in_set)
        #search the element with min distance to all the other elements
        # (the computation can be speeded up, some distance computations are done twice )
        min_dist = np.inf
        min_el = []

        #dist_mat = np.zeros((min_el, min_el))
        for i in range(n_el):
            curr_dist = 0
            for j in range(n_el):
                if i != j:
                    #compute distance
                    curr_dist = curr_dist + np.linalg.norm(in_set[i]-in_set[j],ord=distance_type)

            if curr_dist < min_dist:
                min_el = in_set[i]
                min_dist = curr_dist
        return min_el


    #Scores the found clusters if labels are avalible with a adapted homogenity score and a compactness score
    #
    #Homogenity score (in [0,1]-> 1 is the best): addapted from (https://www.aclweb.org/anthology/D07-1043.pdf) (we don't have grount truth label so we use most comon one instead)
    #h_s= 1/N sum((Cl_most/Cl_all)) where N is number of clusters/sets, Cl_most the number of the most common class in the set and Cl_all the total number in the set
    #
    #compactness score (in [0,inf]-> 0 is the best): is simple how far away from the true number of clusters
    #c_s=(abs(TN-N)/TN) where TN is true number of clusters and N number of clusters
    def score_clusters(self, known_class = True):

        Z_sys_is=self.Z_sys_is
        G1=self.G1
        N=len(Z_sys_is)

        all_labels=[]
        W_z_h_s=[]
        count_removed_nodes = [0, 0]
        correct_pairs = [0, 0]
        incorrect_pairs = [0, 0]
        count_visited = 0

        for W_z in Z_sys_is:
            W_z_labels=[]
            if len(W_z)>0:
                for w in W_z:
                    count_visited = count_visited + 1
                    current_node = G1.nodes[w]

                    t_class_l=current_node['class_l']
                    W_z_labels.append(t_class_l)
                    all_labels.append(t_class_l)

                    #get pair spec
                    pair_spec = current_node['pair_spec']
                    lsr_node = current_node['idx_lsr']

                    if lsr_node >= 0 and len(pair_spec) > 0:
                        action = pair_spec[0]
                        pair_node_idx = pair_spec[1]

                        #get the node of the second state
                        pair_node = G1.nodes[pair_node_idx]
                        pair_lsr_node = pair_node['idx_lsr']
                        if pair_lsr_node < 0:
                            #the node ofthe second state has been pruned
                            count_removed_nodes[action] = count_removed_nodes[action] + 1
                            #print("pruned second")
                        else:
                            if action == 0 and pair_lsr_node == lsr_node:
                                correct_pairs[action] = correct_pairs[action] + 1
                            elif action == 0 and pair_lsr_node != lsr_node:
                                incorrect_pairs[action] = incorrect_pairs[action] + 1
                            elif action == 1 and pair_lsr_node != lsr_node:
                                correct_pairs[action] = correct_pairs[action] + 1
                            elif action == 1 and pair_lsr_node == lsr_node:
                                incorrect_pairs[action] = incorrect_pairs[action] + 1

                    elif lsr_node < 0 and len(pair_spec) > 0:
                        #the node has been pruned
                        action = pair_spec[0]
                        count_removed_nodes[action] = count_removed_nodes[action] + 1

                if known_class:
                    cl_most_l = max(W_z_labels, key=W_z_labels.count)

                    cl_most = W_z_labels.count(cl_most_l)
                    cl_all=len(W_z_labels)
                    W_z_h_s.append(float(cl_most/cl_all))

        if not self.directed_graph and known_class:
            h_s = 0
            if N > 0:
                h_s=sum(W_z_h_s)/N
            TN=len(set(all_labels))
            c_s = -1
            if TN > 0:
                c_s=(float(abs(TN-N)/TN))

            self.h_s=h_s
            self.c_s=c_s

        else:
            h_s=-1
            c_s=-1

        # print("Removed pairs: "+ str(count_removed_nodes))
        # print("Correct pairs: "+str(correct_pairs))
        # print("Incorrect pairs: "+str(incorrect_pairs))
        # print("Tot act: "+str(self.action_count))
        # print("Tot no act: "+str(self.no_action_count))
        # print("Len z all: "+str(len(self.Z_all_data))+", tot count: "+str(self.action_count+self.no_action_count))
        # print("count_visited "+str(count_visited))

        correct_noac = correct_pairs[0]/self.no_action_count
        correct_ac = correct_pairs[1]/self.action_count

        if not self.directed_graph:
            S = [self.G2.subgraph(c).copy() for c in nx.connected_components(self.G2)]
        if self.directed_graph:
            S = [self.G2.subgraph(c).copy() for c in nx.weakly_connected_components(self.G2)]

        max_num_nodes=-1
        max_num_edges=-1
        nodes_conn_comp = []
        edges_conn_comp = []
        for g in S:
            nodes_conn_comp.append(g.number_of_nodes())
            edges_conn_comp.append(g.number_of_edges())
            if g.number_of_nodes()>max_num_nodes:
                max_num_nodes=g.number_of_nodes()
            if g.number_of_edges()>max_num_edges:
                max_num_edges=g.number_of_edges()




        #action performance:
        if known_class:
            if self.directed_graph:
                err_lsr,err_lsr_path, action_error =self.action_score(self.latent_map_file,task="folding")
            else:
                err_lsr,err_lsr_path, action_error =self.action_score(self.latent_map_file,task="stacking")
        else:
            err_lsr = -1
            err_lsr_path = -1
            action_error = -1
            h_s = -1
            c_s = -1
            if self.directed_graph:
                err_lsr,err_lsr_path, action_error =self.action_score(self.latent_map_file,task="folding")
            else:
                err_lsr,err_lsr_path, action_error =self.action_score(self.latent_map_file,task="stacking")


        info_score = dict()
        info_score["removed_pairs"] = count_removed_nodes
        info_score["count_visited"] = count_visited
        info_score["correct_pairs"] = correct_pairs
        info_score["incorrect_pairs"] = incorrect_pairs
        info_score["no_action_count"] = self.no_action_count
        info_score["action_count"] = self.action_count
        info_score["num_conn_comp"] = len(S)
        info_score["nodes_conn_comp"] = nodes_conn_comp
        info_score["max_nodes_comp"] = max_num_nodes
        info_score["edges_conn_comp"] = edges_conn_comp
        info_score["max_edges_comp"] = max_num_edges
        info_score["1 move error on LSR"] = err_lsr
        info_score["1 move error on LSR path to long"] = err_lsr_path
        info_score["action error on 1 move"] = action_error



        return [h_s, c_s, correct_noac, correct_ac, action_error, info_score ]

    def action_score(self,latent_map_file,task="stacking"):
        f = open(latent_map_file, 'rb')
        latent_map = pickle.load(f)
        lsr_count=0
        lsr_fail_count=0
        lsr_fail_count_path=0
        sq_error=[]
        for latent_pair in latent_map:
            z_pos_c1=latent_pair[0]
            z_pos_c2=latent_pair[1]
            action=latent_pair[2]
            a_lambda=latent_pair[3]
            #we can only consider action cases
            if action==1:
                lsr_count+=1
                c1_close_idx,c2_close_idx=self.get_closest_nodes_in_G(z_pos_c1,z_pos_c2,self.G2)
                #check if same node
                if c1_close_idx==c2_close_idx:
                    lsr_fail_count+=1
                else:
                    #chekc the path existis
                    path_exist=nx.has_path(self.G2, source=c1_close_idx, target=c2_close_idx)
                    if not path_exist:
                        lsr_fail_count+=1
                    else:
                        #get the path
                        paths=nx.all_shortest_paths(self.G2, source=c1_close_idx, target=c2_close_idx)
                        for path in paths:
                            #check that it is only on epath
                            if len(path) >2:
                                #print(path)
                                lsr_fail_count_path+=1
                            else:
                                if task=="stacking":
                                    p_lambda=self.G2.edges[path[0], path[1]]['t_lambda']
                                if task=="folding":
                                    p_lambda=self.G2.edges[path[0], path[1]]['t_lambda']
                                    #fix stupid format!
                                    p_lambda=np.array([p_lambda[0][0],p_lambda[0][1],p_lambda[1][0],p_lambda[1][1],p_lambda[2]])
                                    a_lambda=np.array([a_lambda[0][0],a_lambda[0][1],a_lambda[1][0],a_lambda[1][1],a_lambda[2]])
                                sq_error.append(np.square(p_lambda-a_lambda))


        err_lsr=float(lsr_fail_count)/float(lsr_count)
        err_lsr_path=float(lsr_fail_count_path)/float(lsr_count)
        return err_lsr,err_lsr_path,np.mean(sq_error)







    #LSR phase 1 defaul*****************************************************
    # INPUT:
    #   latent_map_file - pkl with encoded samples
    #   distance_type - L1, L2, or Linf distance
    #   directed_graph = False
    #   verbose = False
    # OUTPUT:
    #   G1 - graph to preserve the action conection and have the right format for nodes
    #   Z_all - all the encoded datapoints
    def lsr_phase_1(self, latent_map_file, directed_graph):
        verbose = self.verbose
        distance_type = self.distance_type
        # load latent data
        f = open(latent_map_file, 'rb')
        latent_map = pickle.load(f)
        len_latent_map = len(latent_map)
        #Build the Graph
        #Phase 1 ***************************************
        if directed_graph:
            G1 = nx.DiGraph()
            G2 = nx.DiGraph()
        else:
            G1=nx.Graph()
            G2=nx.Graph()
        self.G2 = G2
        #1.1 build all nodes
        counter=0
        Z_all=set()
        Z_all_data = []
        action_count = 0
        no_action_count = 0
        for latent_pair in latent_map:
            counter+=1
            if verbose:
                print("checking " + str(counter)+ " / " + str(len_latent_map)+ " build " + str(G1.number_of_nodes()) + " so far.")


            # get the latent coordinates
            z_pos_c1=latent_pair[0]
            z_pos_c2=latent_pair[1]
            action=latent_pair[2]
            c1_class_label=latent_pair[4]
            c2_class_label=latent_pair[5]

            #action pairs
            dis=np.linalg.norm(z_pos_c1-z_pos_c2,ord=distance_type)
            if action==1:
                a_lambda=np.array(latent_pair[3])
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c1, idx_lsr = -1, visited = 0,class_l=c1_class_label, pair_spec = (action, c_idx + 1))
                Z_all.add(c_idx)
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c2, idx_lsr = -1, visited = 0,class_l=c2_class_label, pair_spec = ())
                Z_all.add(c_idx)
                G1.add_edge(c_idx-1,c_idx,l=np.round(dis,1),a_lambda=a_lambda)
                action_count = action_count + 1
            #no action
            if action==0:# and dis<epsilon:
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c1, idx_lsr = -1, visited = 0,class_l=c1_class_label, pair_spec = (action, c_idx + 1))
                Z_all.add(c_idx)
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c2, idx_lsr = -1, visited = 0,class_l=c2_class_label, pair_spec = ())
                Z_all.add(c_idx)
                no_action_count = no_action_count + 1

            Z_all_data.append(z_pos_c1)
            Z_all_data.append(z_pos_c2)
        if verbose:
            #Print result of phase 1
            print("***********Phase one done*******")
            print("Num nodes: " + str(G1.number_of_nodes()))
            print("Num edges: " + str(G1.number_of_edges()))
            print("num in Z_all: " + str(len(Z_all)) )

        self.G1 = G1
        self.Z_all = Z_all
        self.Z_all_data = Z_all_data
        self.action_count = action_count
        self.no_action_count = no_action_count

    #LSR phase 2 defaul*****************************************************
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
        distance_type = self.distance_type
        H1 = self.G1.copy()
        Z_all = self.Z_all
        epsilon = self.epsilon
        Z_sys_is=[]
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
            #W_w_to_check=W_z
            while not s_len_wz == e_len_wz:
                s_len_wz=len(W_z)
                #2.2 find all w E G1 for wich ||z-w||_d < 2*epsilon
                W_w=set()
                for w in W_w_to_check:
                    w_pos=H1.nodes[w]['pos']
                    for wn in H1:
                        wn_pos=H1.nodes[wn]['pos']
                        dis=np.linalg.norm(wn_pos-w_pos,ord=distance_type)
                        #could be smaler to be more conservative
                        if dis < epsilon:
                            W_w.add(wn)
                #check speedup
                for w in W_w_to_check:
                    H1.remove_node(w)
                W_w_to_check=W_w-W_w_to_check
                W_z=W_z.union(W_w)
                e_len_wz=len(W_z)
                #for speedup remove W_z from G1 copy
            #2.4 Z_noaction:=Z_noaztipn - W_z
            Z_all = Z_all - W_z
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

    def get_LSR_node_pos(self, w_pos_all, W_z, g_idx):
        if self.use_medoid:
            W_z_c_pos = self.compute_medoid(w_pos_all)
        else:
            W_z_c_pos = np.mean(w_pos_all,axis=0)
            #check if it is in component
            in_z_sys=False
            dis_min_idx=-1
            dis_min=np.inf

            print("********* I am here: get_LSR_node_pos")
            for w in W_z:
                dis=np.linalg.norm(W_z_c_pos-self.G1.nodes[w]['pos'], ord = self.distance_type)
                if dis < self.epsilon:
                    in_z_sys=True
                if dis<dis_min:
                    dis_min=dis
                    dis_min_idx=w

            if not in_z_sys:
                W_z_c_pos=self.G1.nodes[dis_min_idx]['pos']

        return W_z_c_pos

    def get_LSR_node_pos_default(self, w_pos_all, W_z, g_idx):
        if self.use_medoid:
            W_z_c_pos = self.compute_medoid(w_pos_all)
        else:
            W_z_c_pos = np.mean(w_pos_all,axis=0)

        return W_z_c_pos

    def lsr_phase_3p1_nodes(self, config_file,checkpoint_file, decode_img):
        if decode_img:
             #load VAE to decode image if required
            vae_config_file = os.path.join('.', 'configs', config_file + '.py')
            vae_directory = os.path.join('.', 'models', checkpoint_file)

            vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config
            vae_config['exp_name'] = config_file
            vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
            vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])

            vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
            vae_algorithm.model.eval()
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Phase 3 ********************************************************
        Z_sys_is = self.Z_sys_is
        G1 = self.G1
        G2 = self.G2
        verbose = self.verbose
        distance_type = self.distance_type
        #3.1 build centroids-W_z nodes
        for W_z in Z_sys_is:
            w_pos_all = []
            idx_all = []
            c_idx = G2.number_of_nodes()
            if len(W_z)>0:
                for w in W_z:
                    w_pos=G1.nodes[w]['pos']
                    w_pos_all.append(w_pos)
                    idx_all.append(G1.nodes[w]['idx'])
                    G1.nodes[w]['idx_lsr'] = c_idx


                W_z_c_pos = self.get_LSR_node_pos(w_pos_all, W_z, c_idx)


                #decode image
                if decode_img:
                    z_pos=torch.from_numpy(W_z_c_pos).float().to(device)
                    z_pos=z_pos.unsqueeze(0)
                    img_rec,_=vae_algorithm.model.decoder(z_pos)
                    img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
                    G2.add_node(c_idx,pos=W_z_c_pos,image=img_rec_cv,W_z=W_z,w_pos_all=w_pos_all, idx_all = idx_all)
                else:
                    G2.add_node(c_idx,pos=W_z_c_pos,W_z=W_z,w_pos_all=w_pos_all, idx_all = idx_all)

        self.G1 = G1
        self.G2 = G2

    def lsr_phase_3p2_edges(self,a_lambda_format):
        #3.2 build edges
        G1 = self.G1
        G2 = self.G2
        distance_type = self.distance_type
        verbose = self.verbose
        for g2 in G2:
            if verbose:
                print(str(g2)+ " / " + str(G2.number_of_nodes()))
            #W_z hold components of nodes
            W_z=G2.nodes[g2]['W_z']
            #for each component
            for w in W_z:
                #find the partner
                w_pairs=G1.neighbors(w)
                for w_pair in w_pairs:
                    neig_lsr_idx = G1.nodes[w_pair]['idx_lsr']
                    if neig_lsr_idx >= 0:
                        dis=np.linalg.norm(G2.nodes[neig_lsr_idx]['pos']-G2.nodes[g2]['pos'],ord=distance_type)
                        if not G2.has_edge(g2,neig_lsr_idx):

                            if a_lambda_format == None:
                                G2.add_edge(g2,neig_lsr_idx,l=np.round(dis,1),ew=1)
                            else:
                                #read out a_lambda from edge of G1
                                a_lambda=G1.edges[w, w_pair]['a_lambda']
                                G2.add_edge(g2,neig_lsr_idx,l=np.round(dis,1),ew=1,t_lambda=a_lambda)

                            if verbose:
                                print("Num edges: "+str(G2.number_of_edges()))
                        else:
                            #update edge
                            if a_lambda_format == None:
                                ew=G2.edges[g2, neig_lsr_idx]['ew']
                                l=G2.edges[g2, neig_lsr_idx]['l']
                                G2.edges[g2, neig_lsr_idx]['l']=(ew*l+dis)/(ew+1)
                                ew+=1
                                G2.edges[g2, neig_lsr_idx]['ew']=ew
                            else:
                                #simple avarage should work on all formats so far ... we can add more with specific flags like 'stacking'
                                a_lambda=G1.edges[w, w_pair]['a_lambda']
                                g_lambda=G2.edges[g2, neig_lsr_idx]['t_lambda']
                                ew=G2.edges[g2, neig_lsr_idx]['ew']
                                #take simple weighted average
                                t_lambda=(ew*g_lambda+a_lambda)/(ew+1)
                                G2.edges[g2, neig_lsr_idx]['ew']=ew+1
                                G2.edges[g2, neig_lsr_idx]['t_lambda']=t_lambda


        self.G2 = G2



    #LSR phase 3*****************************************************
    # INPUT:
    #   G1 - the first graph
    #   Z_sys_is - cluster with asigned nodes
    #   epsilon - epsilon parameter
    #   distance_type - L1, L2, or Linf distance
    #   config_file - vae config file
    #   checkpoint_file - VAE checkpoint file
    #   verbose = False
    # OUTPUT:
    #   G2 - LSR
    def lsr_phase_3(self, config_file, checkpoint_file, a_lambda_format, decode_img = False ):
        self.lsr_phase_3p1_nodes(config_file, checkpoint_file, decode_img)
        self.lsr_phase_3p2_edges(a_lambda_format)
        if self.verbose:
            print("***********Phase three done*******")
            print("Num nodes: " + str(self.G2.number_of_nodes()))
            print("Num edges: " + str(self.G2.number_of_edges()))




    #LSR phase 4 *****************************************************
    # INPUT:
    #   G2 - LSR
    #   graph_name - name of graph
    #   min_edge_w=0
    #   min_node_m=0
    #   save_graph = False
    #   verbose = False
    # OUTPUT:
    #   G2 - Pruned LSR
    #   stats
    def lsr_phase_4(self, graph_name, min_edge_w , min_node_m, save_graph):
        #phase 4 Pruning
        G2 = self.G2
        verbose = self.verbose

        if verbose:
            print("Pruning edges with ew < " + str(min_edge_w))
        num_edges=G2.number_of_edges()
        remove_edges=[]
        for edge in G2.edges:
            sidx=edge[0]
            gidx=edge[1]
            ew=G2.edges[sidx, gidx]['ew']
            if ew < min_edge_w:
                remove_edges.append((sidx,gidx))
        for re in remove_edges:
            G2.remove_edge(re[0],re[1])
        num_edges_p=G2.number_of_edges()
        if verbose:
            if num_edges > 0:
                print("pruned " + str(num_edges-num_edges_p) + " edges ( " + str(100-(num_edges_p*100.)/num_edges) + " %")
            else:
                print("pruning: num_edges = 0")

        #pruine weak nodes
        if verbose:
            print("Pruning nodes with mearges < " + str(min_node_m))
        num_nodes=G2.number_of_nodes()
        remove_nodes=[]
        for g in G2.nodes:
            ngm=G2.nodes[g]['w_pos_all']
            if len(ngm) < min_node_m:
                remove_nodes.append(g)

        for re in remove_nodes:
            for idx in G2.nodes[re]['idx_all']:
                self.G1.nodes[idx]['idx_lsr'] = -1
            G2.remove_node(re)

        num_nodes_p=G2.number_of_nodes()
        if verbose:
            if num_nodes > 0:
                print("pruned " + str(num_nodes-num_nodes_p) + " nodes ( " + str(100-(num_nodes_p*100.)/num_nodes) + " %")

        #prune single nodes
        num_nodes=G2.number_of_nodes()
        remove_nodes=[]
        isolates=nx.isolates(G2)
        for iso in isolates:
            remove_nodes.append(iso)

        for re in remove_nodes:
            for idx in G2.nodes[re]['idx_all']:
                self.G1.nodes[idx]['idx_lsr'] = -1
            G2.remove_node(re)

        if self.keep_biggest_component:
            S = [G2.subgraph(c).copy() for c in sorted(nx.connected_components(G2), key=len, reverse=True)]
            #print("len s1 "+str(len(S)))
            # if len(S)>0:
            #     max_num_nodes=S[0].number_of_nodes()
            remove_nodes=[]
            for i in range(1,len(S)):
                g = S[i]
                #print(g.number_of_nodes())

                for idx in g:
                    remove_nodes.append(idx)


            # print("max nodes")
            # print(max_num_nodes)
            # print("Num nodes: " + str(G2.number_of_nodes()))

            for re in remove_nodes:
                for idx in G2.nodes[re]['idx_all']:
                    self.G1.nodes[idx]['idx_lsr'] = -1
                G2.remove_node(re)


            # S = [G2.subgraph(c).copy() for c in sorted(nx.connected_components(G2), key=len, reverse=True)]
            # print("len s2: "+str(len(S)))
            # print("Num nodes: " + str(G2.number_of_nodes()))

        if verbose:
            print("pruned " + str(num_nodes-G2.number_of_nodes()) +" isolated nodes")
            print("final Graph ************************************")
            print("Num nodes: " + str(G2.number_of_nodes()))
            print("Num edges: " + str(G2.number_of_edges()))

        if save_graph:
            graph_base_path = "graphs"

            if not os.path.exists(graph_base_path):
                os.mkdir(graph_base_path)

            graph_path = graph_base_path+"/"+graph_name
            nx.write_gpickle(G2, graph_path+".pkl")
            print("SAVED")

        self.stats_dict = {'num_nodes':num_nodes_p}
        self.G2 = G2


    #Find closest nodes in graph G using distance type (1, 2, np.inf)
    def get_closest_nodes(self, z_pos_c1, z_pos_c2):
        distance_type = self.distance_type
        G = self.G1 #the iteration is performed on the graph with all the points
        c1_close_idx=-1
        c2_close_idx=-1
        min_distance_c1=np.Inf
        min_distance_c2=np.Inf

        #find the closest nodes
        for g in G.nodes:
            tz_pos=G.nodes[g]['pos']
            node_distance_c1=np.linalg.norm(z_pos_c1-tz_pos, ord=distance_type)
            node_distance_c2=np.linalg.norm(z_pos_c2-tz_pos, ord=distance_type)
            idx_lsr =  G.nodes[g]['idx_lsr']
            if idx_lsr >=0 and node_distance_c1<min_distance_c1:
                min_distance_c1=node_distance_c1
                c1_close_idx=idx_lsr

            if idx_lsr >=0 and node_distance_c2<min_distance_c2:
                min_distance_c2=node_distance_c2
                c2_close_idx=idx_lsr

        return c1_close_idx, c2_close_idx

    def get_closest_nodes_in_G(self, z_pos_c1, z_pos_c2,G):
        distance_type = self.distance_type
        c1_close_idx=-1
        c2_close_idx=-1
        min_distance_c1=np.Inf
        min_distance_c2=np.Inf

        #find the closest nodes
        for g in G.nodes:
            tz_pos=G.nodes[g]['pos']
            node_distance_c1=np.linalg.norm(z_pos_c1-tz_pos, ord=distance_type)
            node_distance_c2=np.linalg.norm(z_pos_c2-tz_pos, ord=distance_type)
            if node_distance_c1<min_distance_c1:
                min_distance_c1=node_distance_c1
                c1_close_idx=g

            if node_distance_c2<min_distance_c2:
                min_distance_c2=node_distance_c2
                c2_close_idx=g

        return c1_close_idx, c2_close_idx



    def get_kth_nearest_neigh_in_list(self, curr_pos, k, list_idx):

        distance_type = self.distance_type
        idx_closest = -1
        distance_closest = np.Inf

        #find the closest nodes
        i = 0
        n_el = len(list_idx)
        distance_vec = np.zeros(n_el)
        for idx in list_idx:
            pos = self.G1.nodes[idx]['pos']
            node_distance = np.linalg.norm(pos-curr_pos, ord = distance_type)
            distance_vec[i] = node_distance

            # if node_distance > 0 and node_distance < distance_closest:
            #     distance_closest=node_distance
            #     idx_closest = i
            i = i + 1

        distance_vec = np.sort(distance_vec)
        if k == 'prop2':
            k = int(n_el/2)
        elif k == 'prop3':
            k = int(n_el/3)
        elif k == 'prop10':
            k = int(n_el/10)
        if n_el > k:
            distance_closest = distance_vec[k]
        else:
            distance_closest = distance_vec[-1]

        return distance_closest, distance_vec




    def use_fixed_epsilon(self):
        for g in self.G1.nodes:
            self.G1.nodes[g]['eps'] = self.epsilon

    def approximate_epsilon_kth_NN(self, k = 1):
        print('--- k NN with k = '+str(k))
        for g in self.G2.nodes:
            curr_idx_list = self.G2.nodes[g]['idx_all']
            for idx in curr_idx_list:
                #Consider all the nodes in the cluster
                curr_pos = self.G1.nodes[idx]['pos']
                # Get the kth closest node in the cluster
                distance_closest, _ = self.get_kth_nearest_neigh_in_list(curr_pos, k, curr_idx_list)
                self.G1.nodes[idx]['eps'] = distance_closest


    def check_path_connectivity(self,list_idx):
        distance_type = self.distance_type
        intersecting = len(list_idx)*[False]
        i = 0
        for idx in list_idx:

            curr_pos = self.G1.nodes[idx]['pos']
            curr_eps = self.G1.nodes[idx]['eps']
            #check if there is intersection with any of the other epsilon neighbors in the cluster
            k = 0
            while k <len(list_idx) and intersecting[i] == False:
                idx_in = list_idx[k]
                if idx_in != idx:
                    in_pos = self.G1.nodes[idx_in]['pos']
                    in_eps = self.G1.nodes[idx_in]['eps']
                    if np.linalg.norm(curr_pos-in_pos, ord=distance_type)<=(curr_eps+in_eps):

                        intersecting[i] = True
                k = k + 1
            i = i + 1
        return intersecting


    def approximate_epsilon_avg_dist(self,w_ini_std = 0, median_value = False):
        # median_value: if True the median value of the vector is taken, otherwise the mean
        print('--- avg dist with w ini = '+str(w_ini_std))
        distance_type = self.distance_type
        if distance_type==1:
        	metric='cityblock'
        if distance_type==2:
        	metric='euclidean'
        if distance_type==np.inf:
        	metric='chebyshev'

        for g in self.G2.nodes:
            curr_idx_list = self.G2.nodes[g]['idx_all']
            curr_w_pos = self.G2.nodes[g]['w_pos_all']
            all_dist_vec = np.array([])
            for idx in curr_idx_list:
                #Consider all the nodes in the cluster and compute avg and std
                curr_pos = self.G1.nodes[idx]['pos']
                # Get the distances to all nodes in the node
                _, dist_vec = self.get_kth_nearest_neigh_in_list(curr_pos, 1, curr_idx_list)
                if len(dist_vec) > 1:
                    all_dist_vec = np.concatenate((all_dist_vec, dist_vec[1:]), axis = None)
            if len(all_dist_vec) > 0:
                if median_value:
                    avg_dist = np.median(all_dist_vec)
                else:
                    avg_dist = np.mean(all_dist_vec)
                std_dist = np.std(all_dist_vec)
            else:
                avg_dist = 0
                std_dist = 0

            w_std = w_ini_std
            delta_eps = std_dist
            curr_eps = avg_dist + w_std*delta_eps
            if delta_eps < 0.0001:
                delta_eps = 0.0001
            if curr_eps <= 0:
                curr_eps = delta_eps

            single_cluster = False
            while not single_cluster and len(curr_idx_list)>1:
                print('Loop: '+str(curr_eps)+' delta: '+str(delta_eps))
                # Perform clustering
                clustering = DBSCAN(metric = metric, min_samples = 1, eps = curr_eps).fit(curr_w_pos)
                if len(np.unique(clustering.labels_)) == 1:
                    single_cluster = True
                else:
                    curr_eps = curr_eps + delta_eps

            for idx in curr_idx_list:
                self.G1.nodes[idx]['eps'] = curr_eps


            print(str(curr_eps)+' delta: '+str(delta_eps))

    def approximate_iterative_growing(self):

        for g in self.G2.nodes:

            list_idx = self.G2.nodes[g]['idx_all']
            delta_eps_vec = np.zeros(len(list_idx))
            i = 0
            #initialize the epsilon with half of the distance to the NN
            for idx in list_idx:
                curr_pos = self.G1.nodes[idx]['pos']
                closest_dist, _ = self.get_kth_nearest_neigh_in_list(curr_pos, 1, list_idx)
                self.G1.nodes[idx]['eps'] = closest_dist/2
                delta_eps_vec[i] = closest_dist/10
                i = i +1

            single_cluster = False
            while not single_cluster and len(list_idx)>1:
                # Check connectivity
                list_intersecting = self.check_path_connectivity(list_idx)
                #print(list_intersecting)
                if all(list_intersecting):
                    #all nodes are intersecting
                    single_cluster = True
                else:
                    k = 0
                    for is_intersecting in list_intersecting:

                        if not is_intersecting:
                            #increase epsilon of the node if it is not intersecting
                            idx_to_increase = list_idx[k]
                            self.G1.nodes[idx]['eps'] = self.G1.nodes[idx]['eps'] + delta_eps_vec[k]

                        k = k + 1

    def approximate_epsilon_eps_cluster(self):
        distance_type = self.distance_type
        #format distance types
        if distance_type==1:
        	metric='cityblock'
        if distance_type==2:
        	metric='euclidean'
        if distance_type==np.inf:
        	metric='chebyshev'
        for g in self.G2.nodes:
            curr_w_pos = self.G2.nodes[g]['w_pos_all']
            list_idx = self.G2.nodes[g]['idx_all']
            closest_dist_vec = np.zeros(len(list_idx))
            i = 0
            for curr_pos in curr_w_pos:
                closest_dist, dist_vec = self.get_kth_nearest_neigh_in_list(curr_pos, 1, list_idx)
                closest_dist_vec[i] = closest_dist
                i = i +1

            closest_dist_vec = np.sort(closest_dist_vec)
            delta_eps = closest_dist_vec[0]/10 #(closest_dist_vec[-1]-closest_dist_vec[0])/5
            single_cluster = False
            curr_eps = closest_dist_vec[0]
            while not single_cluster and len(list_idx)>1:
                # Perform clustering
                clustering = DBSCAN(metric = metric,min_samples = 1, eps = curr_eps).fit(curr_w_pos)
                if len(np.unique(clustering.labels_)) == 1:
                    single_cluster = True
                else:
                    curr_eps = curr_eps + delta_eps

            for idx in list_idx:
                self.G1.nodes[idx]['eps'] = curr_eps

    def approximate_epsilon(self, method_tuple):
        # method_tuple: tuple composed of (method string, additional parameters for the method)
        method = method_tuple[0]
        print("Selected approximation method: "+method)
        if method == 'fixed_eps':
            self.use_fixed_epsilon()
        elif method == 'kth_NN':
            self.approximate_epsilon_kth_NN(k = method_tuple[1])
        elif method == 'avg_dist' or  method == 'median_dist':
            self.approximate_epsilon_avg_dist(w_ini_std = method_tuple[1][0], median_value = method_tuple[1][1])
        elif method == 'dbscan_cluster':
            self.approximate_epsilon_eps_cluster()
        elif method == 'iterative_growing':
            self.approximate_iterative_growing()

        if method == 'avg_dist' or  method == 'median_dist' or  method == 'dbscan_cluster':
            self.unique_eps_per_cluster = True
        else:
            self.unique_eps_per_cluster = False


    def clear_valid_regions(self):
        self.eps_approximation_computed = False
        for g in self.G1:
            self.G1.nodes[g]['eps'] = 0

    def get_avg_std_eps(self, method = ('fixed_eps','')):
        if not self.eps_approximation_computed:
            self.approximate_epsilon(method)
            self.eps_approximation_computed = True


        if self.unique_eps_per_cluster:
            #if there is one eps per cluster, the loop is performed on G2
            G_to_consider = self.G2

        else:
            G_to_consider = self.G1
        eps_vec = np.zeros(G_to_consider.number_of_nodes())

        i = 0
        for g in G_to_consider:
            if self.unique_eps_per_cluster:
                list_idx = self.G2.nodes[g]['idx_all']
                #any sample in the cluster can be considered since they have the same epsilon
                node_g1 = list_idx[0]
            else:
                node_g1 = g

            if 'eps' in self.G1.nodes[node_g1]:
                curr_epsilon = self.G1.nodes[node_g1]['eps']
            else:
                curr_epsilon = 0
            eps_vec[i] = curr_epsilon
            i = i +1

        return (np.mean(eps_vec), np.std(eps_vec))


    def check_in_valid_region(self, z_pos, method = ('fixed_eps','')):
        #approximate the epsilon based on the clustering
        if not self.eps_approximation_computed:
            self.approximate_epsilon(method)
            self.eps_approximation_computed = True

        G = self.G1 #the iteration is performed on the graph with all the points
        min_distance_c = np.Inf
        #check if there are nodes with distance < epsilon
        avg_eps = 0
        for g in G.nodes:
            tz_pos = G.nodes[g]['pos']
            node_distance_c = np.linalg.norm(z_pos-tz_pos, ord = self.distance_type)
            idx_lsr =  G.nodes[g]['idx_lsr']
            if 'eps' in G.nodes[g]:
                curr_epsilon = G.nodes[g]['eps']
            else:
                curr_epsilon = 0
            avg_eps = avg_eps + curr_epsilon
            if idx_lsr >=0 and node_distance_c <= curr_epsilon:
                return True, node_distance_c
            elif idx_lsr >=0 and node_distance_c < min_distance_c:
                min_distance_c = node_distance_c

        #print("Avg eps: "+str(avg_eps/G.number_of_nodes()))
        return False, min_distance_c

    def check_edges(self):
        G = self.G2
        edges = G.edges()
        for edge in edges:
            g_lambda=G.edges[edge[0], edge[1]]['t_lambda']

            print(edge)
            print(g_lambda)
            print('----')
