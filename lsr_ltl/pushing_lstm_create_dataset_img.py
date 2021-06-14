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





def check_last_row(path,G):
    # G(((one of the boxes is in lower row) or (X(one of the boxes is in lower row) or (XX(one of the boxes is in lower row) )

    cnt = 0 # number of time the corner cell is empty
    for i,l in enumerate(path):
        # print (i)
        # print (l)
        # print (path[i+1])
        # print (path)
        class_l = G.nodes[l]['_class']



        if np.count_nonzero(class_l[2,:] > 0): # if there is a box in the last row
            cnt =  0
        else:
            cnt += 1
            if cnt >3:
                return 0


    return 1





#plot path
def generate_sequence(length,f_start,f_goal,distance_type,image_save_name,device,vae_algorithm,G,k):

    save_img = 1

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

    # if there is no path return 0,[],0,[] so that this start and goal pair is skipped
    if Paths_number==0:
        return 0,[],0,[]

    # print(len(paths_list[0]))

    # # if the path is too long return 0,[],0,[] so that this start and goal pair is skipped
    # if len(paths_list[0])>length:
    #     return 0,[],0,[]


    print("Number of paths from start to goal: ", Paths_number)

    # random selection of a path index
    idx = random.randint(0,Paths_number-1)
    print("Selected path index: ", idx)



    # path corresponding to idx
    path = paths_list[idx]

    # # make the path length equal to length padding the last elem if necessary
    # while len(path)<length:
    #     path.append(path[-1])

    # print("path: ",path)

    path_img1=[]

    path_z=[]

    for l in path:
        z_pos=G.nodes[l]['pos']
        z_pos = torch.from_numpy(z_pos).float().to(device)
        z_pos = z_pos.unsqueeze(0)
        path_z.append(z_pos)

        img_rec,_=vae_algorithm.model.decoder(z_pos)

        img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
        path_img1.append(img_rec_cv.flatten())

        # print(img_rec_cv.flatten().shape)
        # exit(0)




    # check1 = check_adj(path,G)*check_last_row(path,G)
    check1 = check_last_row(path,G)
    # check1 = check_adj(path,G)








    for idx in range(Paths_number):


        path = paths_list[idx] # selected path

        path_img2=[]


        path_z=[]

        for l in path:
            z_pos=G.nodes[l]['pos']
            z_pos = torch.from_numpy(z_pos).float().to(device)
            z_pos = z_pos.unsqueeze(0)
            path_z.append(z_pos)

            img_rec,_=vae_algorithm.model.decoder(z_pos)

            img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
            path_img2.append(img_rec_cv.flatten())





        # check2 = check_adj(path,G)*check_last_row(path,G)
        check2 = check_last_row(path,G)
        # check2 = check_adj(path,G)
        # print("Check2: ",check2)

        if not(check1==check2):


            return check1,path_img1,check2,path_img2


    return 0,[],0,[]



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



    N = 1000/2# dataset_size/2


    k=0
    cnt2=0
    cnt3=0

    # pos = 0
    # neg = 0

    already_seen_pair2 = []

    already_seen_pair3 = []

    while k<N:

        while True:
            if k%3==2:
                start_idx=random.randint(0,len(dataset2)-1)
                goal_idx=random.randint(0,len(dataset2)-1)
                if not([start_idx,goal_idx] in already_seen_pair2):
                    already_seen_pair2.append([start_idx,goal_idx])

                    i_start=dataset2[start_idx][0]
                    i_goal=dataset2[goal_idx][1]
                    break
            else:
                start_idx=random.randint(0,len(dataset3)-1)
                goal_idx=random.randint(0,len(dataset3)-1)

                if not([start_idx,goal_idx] in already_seen_pair3):
                    already_seen_pair3.append([start_idx,goal_idx])

                    i_start=dataset3[start_idx][0]
                    i_goal=dataset3[goal_idx][1]
                    break








        print("Current number of datapoints: ",k)
        check1,path1,check2,path2 = generate_sequence(length,i_start/255.,i_goal/255.,distance_type,image_save_name,device,vae_algorithm,G,k)


        if check1==check2:
            print("skip")
        else:
            if k%3==2:
                cnt2+=1
            else:
                cnt3+=1

            k = k+1
            # check1
            X.append(path1)
            # check2
            X.append(path2)
            if check1:
                Y.append(1)
                # Y.append([1, 0])
                # pos = pos+1
            else:
                Y.append(0)
                # Y.append([0, 1])
                # neg = neg+1

            if check2:
                Y.append(1)
                # Y.append([1, 0])
                # pos = pos+1
            else:
                Y.append(0)
                # Y.append([0, 1])
                # neg = neg+1

    # print("Total pos: ",pos)
    # print("Total neg: ",neg)
    # print(len(X[0][0]))
    # print(Y)

    print("--finished--")
    print("Number of paths with two boxes: ",cnt2*2)
    print("Number of paths with three boxes: ",cnt3*2)

    with open("./datasets_lstm_img/X.pkl", 'wb') as f:
        pickle.dump(X, f)
    with open("./datasets_lstm_img/Y.pkl", 'wb') as f:
        pickle.dump(Y, f)

    # with open("./datasets_lstm/max_features.pkl", 'wb') as f:
    #     pickle.dump(G.number_of_nodes(), f)

if __name__== "__main__":
  main()
