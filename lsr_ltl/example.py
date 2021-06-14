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



import torch.nn as nn


device = torch.device("cuda")



def pad_seq(s,maxlen,emb_dim):
    padded = np.zeros((maxlen,emb_dim),dtype=np.float32)
    padded[:len(s)] = s
    return list(padded)




class LTLNet(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=.6):
        super(LTLNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True) # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, lens):
        batch_size = x.size(0)
        # x = x.long()

        # print(hidden[0].size())
        # input("Press")

        # print(hidden[0].type())

        # print(x[0][0][0])
        # print(x[1][0][0])


        # # Packs a Tensor containing padded sequences of variable length. If batch_first is True, B x T x * input is expected.
        # x = pack_padded_sequence(x, lens, batch_first = True, enforce_sorted = True) # unpad

        # print(x[0][0][0])
        # print(x[0][1][0])
        # exit(0)

        lstm_out, hidden = self.lstm(x, hidden)

        # # Pads a packed batch of variable length sequences.
        # # It is an inverse operation to pack_padded_sequence().
        # lstm_out, lens2 = pad_packed_sequence(lstm_out, batch_first=True) # pad the sequence to the max length in the batch

        # print(lstm_out)
        # exit(0)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden





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
    paths=nx.all_simple_paths(G, source=c1_close_idx, target=c2_close_idx, cutoff=7)
    # paths=nx.all_shortest_paths(G, source=c1_close_idx, target=c2_close_idx)
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
    all_paths_l=[]
    all_paths_z2=[]

    buffer_img_v=np.ones((f_start.shape[0],30,3),np.uint8)
    buffer_img_tiny=np.ones((f_start.shape[0],5,3),np.uint8)
    path_length=0

    for path in paths_list:
        path_img=[]
        path_img.append(f_start)
        path_img.append(buffer_img_v)
        path_z=[]
        path_l=[]
        path_z2=[]
        path_length=0
        for l in path:
            path_length+=1
            z_pos=G.nodes[l]['pos']
            path_l.append(l)
            path_z2.append(z_pos)
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
        all_paths_l.append(path_l)
        all_paths_z2.append(path_z2)

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

    state_a = './state_dict_adj.pt'
    output_size = 1
    embedding_dim = 64
    hidden_dim = 12
    n_layers = 4

    model_a = LTLNet(output_size, embedding_dim, hidden_dim, n_layers)
    model_a.to(device)
    #Loading the best model
    model_a.load_state_dict(torch.load(state_a))


    state_r = './state_dict_row.pt'

    model_r = LTLNet(output_size, embedding_dim, hidden_dim, n_layers)
    model_r.to(device)
    #Loading the best model
    model_r.load_state_dict(torch.load(state_r))


    for idx in range(Paths_number-1):

        inputs = np.array(pad_seq(all_paths_z2[idx],8,embedding_dim))



        lens =len(all_paths_z[idx])

        h = model_a.init_hidden(1)
        inputs, lens = torch.from_numpy(inputs).to(device), torch.Tensor(lens).to(device)
        inputs = torch.reshape(inputs, (1, 8, embedding_dim))
        # print(inputs)

        # print(inputs.size())
        # exit(0)
        output, h = model_a(inputs, h, lens)

        pred = np.round(output.squeeze().detach().cpu().numpy())

        h = model_r.init_hidden(1)
        output, h = model_r(inputs, h, lens)

        pred2 = np.round(output.squeeze().detach().cpu().numpy())

        if pred==1 and pred2==1:
            print(pred,pred2)
            break

    print("idx", idx)

    a=descale_coords(all_actions[idx])
    print("Actions:\n", a)
    L = all_paths_l[idx]
    # open_file = open("../pybullet/actions/actions_to_run.pkl", "wb")
    # pickle.dump(a, open_file)
    # open_file.close()


    combo_img_vp=[]
    t_path=all_paths_img[idx]
    combo_img_vp.append(np.concatenate([t_path[x] for x in range(len(t_path))],axis=1))
    buffer_img_h=np.ones((30,combo_img_vp[0].shape[1],3),np.uint8)
    combo_img_vp.append(buffer_img_h)

    cv2.imwrite(image_save_name,np.concatenate([combo_img_vp[x] for x in range(len(combo_img_vp)-1)],axis=0))



    return a,L






def main():


    #Example for Latent Space ROadmap on stacking Task
    distance_type=1
    weight=0.5
    config_file="VAE_push_v1"
    checkpoint_file="vae_lastCheckpoint.pth"
    output_file="labeled_latent_spaces/"+config_file+"_latent_space_map"
    dataset_name="push_v1"
    testset_name="push_v12"
    graph_name=config_file  +"_graph"
    action_config="APN_push_L1"
    action_checkpoint_file="apnet_lastCheckpoint.pth"
    image_save_name="./examples/example_variable_1.png"



    f = open('datasets/'+testset_name+'.pkl', 'rb')
    dataset = pickle.load(f)
    # random.seed(rng)
    i_start=cv2.imread('./nodes/1.png',cv2.IMREAD_UNCHANGED)
    i_goal=cv2.imread('./nodes/2.png',cv2.IMREAD_UNCHANGED)


    a1,L1 = make_figure_path_result(i_start/255.,i_goal/255.,'graphs/'+graph_name+'.pkl',config_file,checkpoint_file,action_config,action_checkpoint_file,distance_type,image_save_name)

    i_start=cv2.imread('./nodes/3.png',cv2.IMREAD_UNCHANGED)
    i_goal=cv2.imread('./nodes/4.png',cv2.IMREAD_UNCHANGED)


    image_save_name="./examples/example_variable_2.png"


    a2,L2 = make_figure_path_result(i_start/255.,i_goal/255.,'graphs/'+graph_name+'.pkl',config_file,checkpoint_file,action_config,action_checkpoint_file,distance_type,image_save_name)

    print("--finished--")

    open_file = open("../pybullet/actions/actions_to_run1.pkl", "wb")
    pickle.dump(a1, open_file)
    open_file.close()

    open_file = open("../pybullet/actions/actions_to_run2.pkl", "wb")
    pickle.dump(a2, open_file)
    open_file.close()



    open_file = open("./examples/nodes_indexes1.pkl", "wb")
    pickle.dump(L1, open_file)
    open_file.close()

    open_file = open("./examples/nodes_indexes2.pkl", "wb")
    pickle.dump(L2, open_file)
    open_file.close()

if __name__== "__main__":
  main()
