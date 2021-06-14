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




def descale_coords(x):
    """
    Descales the coordinates from [0, 1] interval back to the original
    image size.
    """
    rescaled = x * np.array([2., 2., 1.0,2.,2.]).astype('float32')
    rounded_coords = np.around(rescaled).astype(int)

    # Filter out of the range coordinates because MSE can be out
    cropped_rounded_coords = np.maximum(0, np.minimum(rounded_coords, 2))
    assert(np.all(cropped_rounded_coords) >= 0)
    assert(np.all(cropped_rounded_coords) <= 2)
    return cropped_rounded_coords.astype(int)

#plot path
def make_figure_path_result(f_start,f_goal,graph_name,config_file,checkpoint_file,action_config,action_checkpoint_file,distance_type,image_save_name):

    #load graph
    G=nx.read_gpickle(graph_name)

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

    #load APN
    ap_config_file = os.path.join('.', 'configs', action_config + '.py')
    ap_directory = os.path.join('.', 'models', action_checkpoint_file)
    ap_config = SourceFileLoader(action_config, ap_config_file).load_module().config
    ap_config['exp_name'] = action_config
    ap_config['model_opt']['exp_dir'] = ap_directory
    ap_algorithm = getattr(alg, ap_config['algorithm_type'])(ap_config['model_opt'])
    ap_algorithm.load_checkpoint('models/'+"APN_push_L1_seed98765"+"/"+action_checkpoint_file)
    ap_algorithm.model.eval()
    print("loaded APN")



    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # paths=nx.all_simple_paths(G, source=c1_close_idx, target=c2_close_idx, cutoff=7)
    paths=nx.all_shortest_paths(G, source=c1_close_idx, target=c2_close_idx)
    # print(list(paths))
    paths_list = list(paths)
    Paths_number = len(paths_list)
    print("Number of paths from start to goal: ", Paths_number)
    if Paths_number==0:
        return 0


    #go to numpy
    f_start=np.squeeze(f_start)
    f_goal=np.squeeze(f_goal)

    all_paths_img=[]
    all_paths_z=[]

    buffer_img_v=np.ones((f_start.shape[0],30,3),np.uint8)
    buffer_img_tiny=np.ones((f_start.shape[0],5,3),np.uint8)
    path_length=0

    for path in paths_list:
        path_img=[]
        path_img.append(f_start)
        path_img.append(buffer_img_v)
        path_z=[]
        path_length=0
        for l in path:
            path_length+=1
            z_pos=G.nodes[l]['pos']

            z_pos = torch.from_numpy(z_pos).float().to(device)
            z_pos = z_pos.unsqueeze(0)
            path_z.append(z_pos)

            img_rec,_=vae_algorithm.model.decoder(z_pos)

            img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
            path_img.append(img_rec_cv)
            path_img.append(buffer_img_tiny)

        path_img = path_img[:-1]
        path_img.append(buffer_img_v)
        path_img.append(f_goal)

        all_paths_img.append(path_img)
        all_paths_z.append(path_z)

    print("all_paths_z size: ",len(all_paths_z))
    #debug visual paths:
    combo_img_vp=[]
    for i in range(len(all_paths_img)):
        t_path=all_paths_img[i]
        combo_img_vp.append(np.concatenate([t_path[x] for x in range(len(t_path))],axis=1))


    #lets get the actions!
    all_actions=[]
    for i in range(Paths_number):
        z_p=all_paths_z[i]
        path_action=[]
        for j in range(len(z_p)-1):
            z1_t=z_p[j]
            z2_t=z_p[j+1]
            action_to=ap_algorithm.model.forward(z1_t,z2_t)
            action=action_to.cpu().detach().numpy()
            action = np.squeeze(action)
            path_action.append(action)
        all_actions.append(path_action)

    print("all_actions size: ",len(all_actions))




    # print(len(all_paths_z))
    #inpainting actions!
    off_x=55
    off_y=80
    len_box=60
    p_color=(1,0,0)
    r_color=(0,1,0)

    cx_vec = [60,130,195,215] # wrt the image coordinates (x positive towards right)
    cy_vec = [35,130,215,215]# wrt the image coordinates (y positive towards down)


    for i in range(Paths_number):
        p_a=all_actions[i]
        p_i=all_paths_img[i]
        img_idx=2
        for j in range(len(p_a)):
            # print("#",j,p_a[j])
            a=descale_coords(p_a[j])
            # print(a)
            t_img=p_i[img_idx]


            px=cx_vec[a[0]]
            py=cy_vec[a[1]]
            cv2.circle(t_img, (px,py), 12, p_color, 4)
            rx=cx_vec[a[3]]
            ry=cy_vec[a[4]]
            cv2.circle(t_img, (rx,ry), 8, r_color, -1)
            all_paths_img[i][img_idx]=t_img
            img_idx+=2

    #make to 255
    for i in range(len(all_paths_img)):
        p_i=all_paths_img[i]
        for j in range(len(p_i)):
            t_img=p_i[j]
            t_img=t_img*255
            t_img_f=t_img.astype("uint8").copy()
            all_paths_img[i][j]=t_img_f


    # print(all_actions[0])

    idx = random.randint(0,Paths_number-1)
    print("idx ",idx)
    a=descale_coords(all_actions[idx])
    print("Actions:\n", a)

    # open_file = open("../pybullet/actions/actions_to_run.pkl", "wb")
    # pickle.dump(a, open_file)
    # open_file.close()


    combo_img_vp=[]
    t_path=all_paths_img[idx]
    combo_img_vp.append(np.concatenate([t_path[x] for x in range(len(t_path))],axis=1))
    buffer_img_h=np.ones((30,combo_img_vp[0].shape[1],3),np.uint8)
    combo_img_vp.append(buffer_img_h)

    cv2.imwrite(image_save_name,np.concatenate([combo_img_vp[x] for x in range(len(combo_img_vp)-1)],axis=0))






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lable_ls' , type=int, default=False, help='Lable latent space')
    parser.add_argument('--build_lsr' , type=int, default=False, help='Build Latent Space Roadmap')
    parser.add_argument('--example' , type=int, default=False, help='Make example path')
    parser.add_argument('--seed', type=int, required=True, default=999,
                    help='random seed')
    args = parser.parse_args()

    #Example for Latent Space ROadmap on stacking Task
    # rng=random.randint(0,9000)
    rng=int(args.seed)
    print("seed: ",rng)
    distance_type=1
    weight=1
    config_file="VAE_push_v1"
    checkpoint_file="vae_lastCheckpoint.pth"
    output_file="labeled_latent_spaces/"+config_file+"_latent_space_map"
    dataset_name="push_v1"
    testset_name="push_v12"
    graph_name=config_file  +"_graph"
    action_config="APN_push_L1"
    action_checkpoint_file="apnet_lastCheckpoint.pth"
    image_save_name="./examples/stacking_example_"+str(rng).zfill(5)+".png"


    #lable latent space
    if args.lable_ls:
        print("labeling latent space")
        lsr.lable_latent_space(config_file,checkpoint_file,output_file,dataset_name)


    #bulid graph
    if args.build_lsr:
        print("Building LSR")
        latent_map_file=output_file+'.pkl'
        mean_dist_no_ac, std_dist_no_ac, dist_list = lsr.compute_mean_and_std_dev(latent_map_file, distance_type,action_mode=0)
        epsilon=mean_dist_no_ac+weight*std_dist_no_ac
        print("epsilon: ",epsilon)
        print("mean_dist_no_ac: ",mean_dist_no_ac)
        print("std_dist_no_ac: ",std_dist_no_ac)
        # epsilon = 0.7
        lsr.build_lsr(latent_map_file,epsilon,distance_type,graph_name, config_file,checkpoint_file,min_edge_w=1,min_node_m=1, directed_graph = False,  hasclasses=False, verbose = False, save_graph = True)

    #select random start and goal state from training set
    #bulid graph
    if args.example:
        print("Generating example")
        f = open('datasets/'+testset_name+'.pkl', 'rb')
        dataset = pickle.load(f)
        # random.seed(rng)
        start_idx=random.randint(0,len(dataset))
        goal_idx=random.randint(0,len(dataset))
        i_start=dataset[start_idx][0]
        i_goal=dataset[goal_idx][1]
        print(dataset[start_idx][4])
        print(dataset[goal_idx][5])
        make_figure_path_result(i_start/255.,i_goal/255.,'graphs/'+graph_name+'.pkl',config_file,checkpoint_file,action_config,action_checkpoint_file,distance_type,image_save_name)

    print("--finished--")




if __name__== "__main__":
  main()
