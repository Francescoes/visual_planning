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
def generate_sequence(length,f_start,f_goal,distance_type,image_save_name,device,vae_algorithm,G,k,X,Y):

    save_img = 0

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

    # # if there is no path return 0,[],0,[] so that this start and goal pair is skipped
    # if Paths_number==0:
    #     return 0,[],0,[]

    # print(len(paths_list[0]))

    # # if the path is too long return 0,[],0,[] so that this start and goal pair is skipped
    # if len(paths_list[0])>length:
    #     return 0,[],0,[]


    print("Number of paths from start to goal: ", Paths_number)

    # random selection of a path index
    for idx in range(Paths_number):

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
            if save_img:
                z_pos = torch.from_numpy(z_pos).float().to(device)
                z_pos = z_pos.unsqueeze(0)
                path_z.append(z_pos)

                img_rec,_=vae_algorithm.model.decoder(z_pos)

                img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
                path_img1.append(img_rec_cv)
                path_img1.append(buffer_img_tiny)

        check1 = check_adj(path,G)

        if save_img:
            path_img1 = path_img1[:-1]
            path_img1.append(buffer_img_v)
            path_img1.append(f_goal)

            #make to 255
            for j in range(len(path_img1)):
                    t_img=path_img1[j]
                    t_img=t_img*255
                    t_img_f=t_img.astype("uint8").copy()
                    path_img1[j]=t_img_f


            combo_img_vp=[]
            combo_img_vp.append(np.concatenate([path_img1[x] for x in range(len(path_img1))],axis=1))
            buffer_img_h=np.ones((30,combo_img_vp[0].shape[1],3),np.uint8)
            combo_img_vp.append(buffer_img_h)

            cv2.imwrite(image_save_name+"_"+str(k)+"_"+str(check1)+".png",np.concatenate([combo_img_vp[x] for x in range(len(combo_img_vp)-1)],axis=0))







        X.append(path1)

        if check1:
            Y.append(1)
            # Y.append([1, 0])
            # pos = pos+1
        else:
            Y.append(0)
            # Y.append([0, 1])
            # neg = neg+1



    return Paths_number







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



    N = 500# dataset_size/2


    k=0


    # total numer of paths
    tot_num=0


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
        paths_num = generate_sequence(length,i_start/255.,i_goal/255.,distance_type,image_save_name,device,vae_algorithm,G,k,X,Y)

        tot_num = tot_num + paths_num

        k += 1



    print("--finished--")
    # print("Number of paths with two boxes: ",cnt2*2)
    # print("Number of paths with three boxes: ",cnt3*2)
    print("Total number of paths: ",tot_num)

    with open("./datasets_lstm/X_a_all.pkl", 'wb') as f:
        pickle.dump(X, f)
    with open("./datasets_lstm/Y_a_all.pkl", 'wb') as f:
        pickle.dump(Y, f)

    # with open("./datasets_lstm/max_features.pkl", 'wb') as f:
    #     pickle.dump(G.number_of_nodes(), f)

if __name__== "__main__":
  main()
