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





def check_corner(path,G):
    # G(i-th box in corner -> not (X(i-th box in conrner))
    pre_number = -1
    cnt = 0
    for i,l in enumerate(path[:-1]):
        # print (i)
        # print (l)
        # print (path[i+1])
        # print (path)
        class_l = G.nodes[l]['_class']



        if not(class_l[2,2] == 0):
            cnt = cnt + 1
            if cnt >2:
                return 0

        else:
            cnt = 0


    return 1





# 0 the box did not move
# 1 the box moved according to rules
# -1 the box moved not according to rules
def did_it_move_correctly(idx_1,idy_1,idx_2,idy_2):
    if (idx_1-idx_2==0) and (idy_1-idy_2==0):
        return 0
    elif (not(idx_1-idx_1==0)) and (not(idy_1-idy_2==0)):
         return -1 # both x and y changed
    else:
        return 1




def is_okay(class1,class2):

    cnt = 0 # number of box which moved

    idx1_1,idy1_1 = np.where(class1==1)
    idx1_2,idy1_2 = np.where(class2==1)

    if list(idx1_1): # if it is not empty
        check = did_it_move_correctly(idx1_1,idy1_1,idx1_2,idy1_2)
        if check==-1:# if the box moved not according the rules
            print("box 1 moved not according the rules")
            return 0
        elif check ==1:
            cnt = cnt +1



    idx2_1,idy2_1 = np.where(class1==2)
    idx2_2,idy2_2 = np.where(class2==2)

    if list(idx2_1): # if it is not empty
        check = did_it_move_correctly(idx2_1,idy2_1,idx2_2,idy2_2)
        if check==-1:# if the box moved not according the rules
            print("box 2 moved not according the rules")
            return 0
        elif check ==1:
            cnt = cnt +1


    idx3_1,idy3_1 = np.where(class1==3)
    idx3_2,idy3_2 = np.where(class2==3)

    if list(idx3_1): # if it is not empty
        check = did_it_move_correctly(idx3_1,idy3_1,idx3_2,idy3_2)
        if check==-1:# if the box moved not according the rules
            print("box 3 moved not according the rules")
            return 0
        elif check ==1:
            cnt = cnt +1


    if cnt >1:
        print("more than 1 box moved")
        return 0
    elif cnt ==0:
        print("nothing changed")
        return 0
    else:
        print("the transition is okay")
        return 1



#plot path
def generate_sequence(length,f_start,f_goal,distance_type,image_save_name,device,vae_algorithm,G):

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
    # paths=nx.all_shortest_paths(G, source=c1_close_idx, target=c2_close_idx)

    # print(list(paths))
    paths_list = list(paths)
    Paths_number = len(paths_list) # number of paths from start to goal

    print("Number of paths from start to goal: ", Paths_number)
    # if there is no path return 0,[],0,[] so that this start and goal pair is skipped
    if Paths_number==0:
        return -1,-1



    valid = 0
    # random selection of a path index
    for idx in range(Paths_number):
        print("Selected path index: ", idx)

        #go to numpy
        f_start=np.squeeze(f_start)
        f_goal=np.squeeze(f_goal)

        buffer_img_v=np.ones((f_start.shape[0],30,3),np.uint8)
        buffer_img_tiny=np.ones((f_start.shape[0],5,3),np.uint8)
        path_length=0

        # path corresponding to idx
        path = paths_list[idx]

        # # make the path length equal to length
        # while len(path)<length:
        #     path.append(path[-1])

        # print("path: ",path)

        path_img=[]
        path_img.append(f_start)
        path_img.append(buffer_img_v)

        path1=[]
        path_z=[]
        path_class=[]
        for l in path:

            z_pos=G.nodes[l]['pos']
            class_pos=G.nodes[l]['_class']
            path1.append(z_pos)
            path_class.append(class_pos)
            if save_img:
                z_pos = torch.from_numpy(z_pos).float().to(device)
                z_pos = z_pos.unsqueeze(0)
                path_z.append(z_pos)

                img_rec,_=vae_algorithm.model.decoder(z_pos)

                img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
                path_img.append(img_rec_cv)
                path_img.append(buffer_img_tiny)

        if save_img:
            path_img = path_img[:-1]
            path_img.append(buffer_img_v)
            path_img.append(f_goal)



        valid1 = 1
        for i in range(len(path_z)-1):
            if not is_okay(path_class[i],path_class[i+1]):
                valid1 = 0

        valid = valid + valid1

        if save_img:
            #make to 255
            for j in range(len(path_img)):
                    t_img=path_img[j]
                    t_img=t_img*255
                    t_img_f=t_img.astype("uint8").copy()
                    path_img[j]=t_img_f


            combo_img_vp=[]
            combo_img_vp.append(np.concatenate([path_img[x] for x in range(len(path_img))],axis=1))
            buffer_img_h=np.ones((30,combo_img_vp[0].shape[1],3),np.uint8)
            combo_img_vp.append(buffer_img_h)

            cv2.imwrite(image_save_name+"_"+str(valid1)+".png",np.concatenate([combo_img_vp[x] for x in range(len(combo_img_vp)-1)],axis=0))
            if valid1 == 0:
                exit(0)


    return valid, Paths_number



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lable_ls' , type=int, default=0, help='Lable latent space')
    parser.add_argument('--build_lsr' , type=int, default=0, help='Build Latent Smake_figure_path_resultpace Roadmap')
    args = parser.parse_args()

    #Example for Latent Space ROadmap on stacking Task
    distance_type=1
    weight=1.0
    config_file="VAE_push_v1"
    checkpoint_file="vae_lastCheckpoint.pth"
    output_file="labeled_latent_spaces/"+config_file+"_latent_space_map"
    dataset_name="push_v1"
    graph_name=config_file  +"_graph"
    label_config="LBN_push_L1"
    label_checkpoint_file="lbnet_lastCheckpoint.pth"

    image_save_name="LSTM_img"





    f = open('datasets/'+dataset_name+'.pkl', 'rb')
    dataset = pickle.load(f)

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

    already_seen_start = []
    already_seen_goal = []


    v = 0 # number of valid paths
    n = 0 # number of paths
    while k<N:

        while True:
            start_idx=random.randint(0,len(dataset)-1)
            goal_idx=random.randint(0,len(dataset)-1)
            if not(start_idx in already_seen_start and goal_idx in already_seen_goal):
                break
        already_seen_start.append(start_idx)
        already_seen_goal.append(goal_idx)

        i_start=dataset[start_idx][0]
        i_goal=dataset[goal_idx][1]

        print("Current number of datapoints: ",k)
        valid,number = generate_sequence(length,i_start/255.,i_goal/255.,distance_type,image_save_name,device,vae_algorithm,G)

        if valid==-1:
            print("skip")
        else:
            v = v + valid
            n = n + number
            k = k+1

    print("number of paths: ",n)
    print("number of valid paths: ",v)
    print("percentage of valid paths: ",v/(n)*100,"%")

if __name__== "__main__":
  main()
