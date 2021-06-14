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





def are_adjacent(class_l):
    if np.count_nonzero(class_l==3):
        x1,y1 = np.where(class_l == 1)
        x2,y2 = np.where(class_l == 2)
        x3,y3 = np.where(class_l == 3)

        if (abs(x1-x2)==1 and abs(y1-y2)==0) or (abs(x1-x2)==0 and abs(y1-y2)==1) or (abs(x1-x3)==1 and abs(y1-y3)==0) or (abs(x1-x3)==0 and abs(y1-y3)==1) or (abs(x2-x3)==1 and abs(y2-y3)==0) or (abs(x2-x3)==0 and abs(y2-y3)==1):
            return 1
        else:
            return 0
    else:
        return 1


def check_adj(path,G):
    # G(((boxes are not adjacent) or (X(boxes are not adjacent) or (XX(boxes are not adjacent) )

    cnt = 0 # number of time the boxes are adjacent
    for i,l in enumerate(path):

        class_l = G.nodes[l]['_class']

        if are_adjacent(class_l):
            cnt += 1

            if cnt > 3:
                return 0 # not valid
        else:
            cnt = 0


    return 1 # valid





#plot path
def generate_sequence(pos_or_neg,length,f_start,f_goal,distance_type,image_save_name,device,vae_algorithm,G,k):


    #get encoding
    f_start=np.expand_dims(f_start,axis=0)
    f_goal=np.expand_dims(f_goal,axis=0)
    #get recon start and goal and z
    x=torch.from_numpy(f_start)
    x=x.float()
    x=x.permute(0,3,1,2)
    x = Variable(x).to(device)
    x2=torch.from_numpy(f_goal)
    x2=x2.float()
    x2=x2.permute(0,3,1,2)
    x2 = Variable(x2).to(device)
    dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(x)
    dec_mean2, dec_logvar2, z2, enc_logvar2=vae_algorithm.model.forward(x2)
    dec_start=dec_mean1[0].detach().permute(1,2,0).cpu().numpy()
    z_start=z[0].cpu().detach().numpy()
    dec_goal=dec_mean2[0].detach().permute(1,2,0).cpu().numpy()
    z_goalt=z2[0].cpu().detach().numpy()

    #get closes start and goal node from graph
    [c1_close_idx, c2_close_idx] = lsr.get_closest_nodes(G, z_start, z_goalt,distance_type)
    #use graph to find paths
    paths=nx.all_simple_paths(G, source=c1_close_idx, target=c2_close_idx, cutoff=length-1)


    # print(list(paths))
    paths_list = list(paths)
    Paths_number = len(paths_list) # number of paths from start to goal

    # if there is no path return -1,[] so that this start and goal pair is skipped
    if Paths_number==0:
        return -1,[]


    print("Number of paths from start to goal: ", Paths_number)

    # random selection of a path index
    idx = random.randint(0,Paths_number-1)
    print("Selected path index: ", idx)

    #go to numpy
    f_start=np.squeeze(f_start)
    f_goal=np.squeeze(f_goal)

    buffer_img_v=np.ones((f_start.shape[0],30,3),np.uint8)
    buffer_img_tiny=np.ones((f_start.shape[0],5,3),np.uint8)

    # path corresponding to idx
    path = paths_list[idx]

    # # make the path length equal to length padding the last elem if necessary
    # while len(path)<length:
    #     path.append(path[-1])

    # print("path: ",path)

    path_img1=[]
    path_img1.append(f_start)
    path_img1.append(buffer_img_v)

    path1=[]
    path_z=[]

    for l in path:
        z_pos=G.nodes[l]['pos']
        path1.append(z_pos)




    check1 = check_adj(path,G)

    if check1 == pos_or_neg:
        return check1,path1
    else:
        return -1,[]




def main():

    #Example for Latent Space ROadmap on stacking Task
    config_file="VAE_push_v1"
    checkpoint_file="vae_lastCheckpoint.pth"
    output_file="labeled_latent_spaces/"+config_file+"_latent_space_map"

    graph_name=config_file  +"_graph"
    distance_type = 1

    image_save_name="./Img_lstm/LSTM_img"




    print("Generate dataset for LSTM training")
    dataset_name2="push_v12"
    f = open('datasets/'+dataset_name2+'.pkl', 'rb')
    dataset2 = pickle.load(f)
    dataset_name3="push_v13"
    f = open('datasets/'+dataset_name3+'.pkl', 'rb')
    dataset3 = pickle.load(f)

    X = [] # list of sequences
    Y = [] # list of sequences
    length = 8 # sequences length

    #load graph
    G = nx.read_gpickle('graphs/'+graph_name+'.pkl')


    #load VAE
    vae_config_file = os.path.join('.', 'configs', config_file + '.py')
    vae_directory = os.path.join('.', 'models', checkpoint_file)
    vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config
    vae_config['exp_name'] = config_file
    vae_config['vae_opt']['exp_dir'] = vae_directory
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
    vae_algorithm.model.eval()
    print("loaded VAE")

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    N = 4000# dataset_size/3


    k=0
    cnt2=0
    cnt3=0


    pos = 0
    neg = 0

    already_seen_pair2 = []

    already_seen_pair3 = []



    N1 = 267
    while k<N1:
        print("Current number of datapoints 2 boxes: ",k)

        start_idx=random.randint(0,len(dataset2)-1)
        goal_idx=random.randint(0,len(dataset2)-1)
        i_start=dataset2[start_idx][0]
        i_goal=dataset2[goal_idx][1]

        # while True:
        #
        #     start_idx=random.randint(0,len(dataset2)-1)
        #     goal_idx=random.randint(0,len(dataset2)-1)
        #     if not([start_idx,goal_idx] in already_seen_pair2):
        #         already_seen_pair2.append([start_idx,goal_idx])
        #
        #         i_start=dataset2[start_idx][0]
        #         i_goal=dataset2[goal_idx][1]
        #         break

        check,path = generate_sequence(1,length,i_start/255.,i_goal/255.,distance_type,image_save_name,device,vae_algorithm,G,k)

        if check == -1:
            print("skip")
        else:
            cnt2+=1
            k = k+1

            X.append(path)
            Y.append(1)
            pos +=1


    N2 = 3738
    k = 0
    N3 = int(75/100*N2)

    while k<N2:
        print("Current number of datapoints 3 boxes: ",k)

        if k < N3:
            pos_or_neg = 1
        else:
            pos_or_neg = 0


        start_idx=random.randint(0,len(dataset3)-1)
        goal_idx=random.randint(0,len(dataset3)-1)
        i_start=dataset3[start_idx][0]
        i_goal=dataset3[goal_idx][1]

        # while True:
        #     start_idx=random.randint(0,len(dataset3)-1)
        #     goal_idx=random.randint(0,len(dataset3)-1)
        #     if not([start_idx,goal_idx] in already_seen_pair3):
        #         already_seen_pair3.append([start_idx,goal_idx])
        #
        #         i_start=dataset3[start_idx][0]
        #         i_goal=dataset3[goal_idx][1]
        #         break

        check,path = generate_sequence(pos_or_neg,length,i_start/255.,i_goal/255.,distance_type,image_save_name,device,vae_algorithm,G,k)

        if check == -1:
            print("skip")
        else:
            X.append(path)
            Y.append(pos_or_neg)

            cnt3+=1
            k+=1
            if k < N3:
                pos +=1
            else:
                neg +=1








    print("--finished--")
    print("Number of paths with two boxes: ",cnt2)
    print("Number of paths with three boxes: ",cnt3)
    print("Number of valid paths: ",pos)
    print("Number of non-valid paths: ", neg)

    with open("./datasets_lstm_asym/X_a.pkl", 'wb') as f:
        pickle.dump(X, f)
    with open("./datasets_lstm_asym/Y_a.pkl", 'wb') as f:
        pickle.dump(Y, f)

    # with open("./datasets_lstm/max_features.pkl", 'wb') as f:
    #     pickle.dump(G.number_of_nodes(), f)

if __name__== "__main__":
  main()
