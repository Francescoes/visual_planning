#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:20:51 2020

@author: petrapoklukar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class PrintShape(nn.Module):
    def __init__(self, message):
        super(PrintShape, self).__init__()
        self.message = message
        
    def forward(self, feat):
#        print(self.message, feat.shape)
        return feat
    
class MaskedSigmoid(nn.Module):
    def __init__(self):
        super(MaskedSigmoid, self).__init__()
        self.mask = torch.Tensor([0., 0., 1., 0., 0.])
        self.opp_mask = torch.Tensor([1., 1., 0., 1., 1.])
        
    def forward(self, feat):
        sig_feat = nn.Sigmoid()(feat) * self.opp_mask
        res = self.mask * feat + sig_feat
        return res
    

class APNet_Mlp_1Head_2Bodies_1Tail(nn.Module):
    """
    Action proposal network with Linear layers.
    
    Input: latent sample from a VAE of dim latent_dim.
    Output: input_uv, input_h, output_uv
    """
    def __init__(self, opt, trained_params=None):
        super().__init__()
        self.opt = opt
        self.shared_imgnet_dims = opt['shared_imgnet_dims'] # [128, 64],
        self.separated_imgnet_dims = opt['separated_imgnet_dims'] # [64, 32, 16]
        self.shared_repr_dims = opt['shared_reprnet_dims'] # [16*2, 16, 8, 5]
            
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        # After training parameters for getting the right coords
        if trained_params is not None:
            self.data_max = trained_params['data_max']
            self.data_min = trained_params['data_min']
            self.norm_mean = trained_params['norm_mean']
            self.norm_std = trained_params['norm_std']
            
        
        # --- Shared input img & output img network
        self.shared_imgnet = nn.Sequential()
        self.shared_imgnet.add_module('shared_imgnet_shape0', PrintShape('Shared ImgNet Input 0'))
        for i in range(len(self.shared_imgnet_dims) - 1):
        
            self.shared_imgnet.add_module('shared_imgnet_lin' + str(i), nn.Linear(
                    self.shared_imgnet_dims[i], self.shared_imgnet_dims[i+1]))    
            self.shared_imgnet.add_module('shared_imgnet_relu' + str(i), nn.ReLU())
            self.shared_imgnet.add_module('shared_imgnet_shape' + str(i+1), PrintShape(
                'Shared ImgNet Input ' + str(i+1)))
        
        # --- Separate input img & output img network
        self.input_net = nn.Sequential()
        self.output_net = nn.Sequential()
        
        self.input_net.add_module('input_net_shape0' , PrintShape('Input Net 0'))
        self.output_net.add_module('output_net_shape0', PrintShape('Output Net 0'))
        for i in range(len(self.separated_imgnet_dims) - 1):
        
            self.input_net.add_module('input_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.input_net.add_module('input_net_relu' + str(i), nn.ReLU())
            self.input_net.add_module('input_net_shape' + str(i+1), PrintShape(
                'Input Net ' + str(i+1)))
            
            self.output_net.add_module('output_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.output_net.add_module('output_net_relu' + str(i), nn.ReLU())    
            self.output_net.add_module('output_net_shape' + str(i+1), PrintShape(
                'Output Net ' + str(i+1)))
        
        # --- Shared network
        self.shared_reprnet = nn.Sequential()
        self.shared_reprnet.add_module('shared_reprnet_shape0', PrintShape('Shared Net 0'))
        for i in range(len(self.shared_repr_dims) - 1):  
            self.shared_reprnet.add_module('shared_reprnet_lin' + str(i), nn.Linear(
                    self.shared_repr_dims[i], self.shared_repr_dims[i+1]))
            if i != len(self.shared_repr_dims) - 2:
                self.shared_reprnet.add_module('shared_reprnet_relu' + str(i), nn.ReLU())
            self.shared_reprnet.add_module('shared_reprnet_shape' + str(i+1), PrintShape(
                    'Shared Net ' + str(i+1)))
#        self.shared_reprnet.add_module('sigmoid', nn.Sigmoid())
#        self.shared_reprnet.add_module('masked_sigmoid', MaskedSigmoid())
        
        self.weight_init()
    
    def weight_init(self):
        """
        Weight initialiser.
        """
        initializer = globals()[self.opt['weight_init']]

        for block in self._modules:
            b = self._modules[block]
            if isinstance(b, nn.Sequential):
                for m in b:
                    initializer(m)
            else:
                initializer(b)
                
    def forward(self, img1, img2):
        img1_interrepr = self.shared_imgnet(img1)
        img1_repr = self.input_net(img1_interrepr)
        
        img2_interrepr = self.shared_imgnet(img2)
        img2_repr = self.output_net(img2_interrepr)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        out = self.shared_reprnet(concat_repr)
        return out
    
    def descale_coords(self, x):
        rescaled = x * (self.data_max - self.data_min) + self.data_min
        rounded_coords = np.around(rescaled).astype(int)
        # Filter out of the range coordinates
        cropped_rounded_coords = np.maximum(self.data_min, np.minimum(rounded_coords, self.data_max))
        assert(np.all(cropped_rounded_coords) >= self.data_min)
        assert(np.all(cropped_rounded_coords) <= self.data_max)
        return cropped_rounded_coords.astype(int)
    
    def denormalise(self, x):
        denormalised = x.detach().cpu().numpy() * self.norm_std + self.norm_mean
        assert(np.all(denormalised) >= 0.)
        assert(np.all(denormalised) <= 1.)
        return denormalised
    
    def forward_and_transform(self, img1, img2):
        img1_interrepr = self.shared_imgnet(img1)
        img1_repr = self.input_net(img1_interrepr)
        
        img2_interrepr = self.shared_imgnet(img2)
        img2_repr = self.output_net(img2_interrepr)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        out = self.shared_reprnet(concat_repr)
        
        out_denorm = self.denormalise(out)
        out_denorm_height = out_denorm[:, 2]
        out_descaled = self.descale_coords(out_denorm)
        out_descaled[:, 2] = out_denorm_height
        return out_descaled



class APNet_Mlp_1H2B1T_OneHotToy(nn.Module):
    """
    WIP for binary classification
    """
    def __init__(self, opt, trained_params=None):
        super().__init__()
        self.opt = opt
        self.shared_imgnet_dims = opt['shared_imgnet_dims'] # [128, 64],
        self.separated_imgnet_dims = opt['separated_imgnet_dims'] # [64, 32, 16]
        self.shared_repr_dims = opt['shared_reprnet_dims'] # [16*2, 16, 8, 5]
            
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        # --- Shared input img & output img network
        self.shared_imgnet = nn.Sequential()
        self.shared_imgnet.add_module('shared_imgnet_shape0', PrintShape('Shared ImgNet Input 0'))
        for i in range(len(self.shared_imgnet_dims) - 1):
        
            self.shared_imgnet.add_module('shared_imgnet_lin' + str(i), nn.Linear(
                    self.shared_imgnet_dims[i], self.shared_imgnet_dims[i+1]))    
            self.shared_imgnet.add_module('shared_imgnet_relu' + str(i), nn.ReLU())
            self.shared_imgnet.add_module('shared_imgnet_shape' + str(i+1), PrintShape(
                'Shared ImgNet Input ' + str(i+1)))
        
        # --- Separate input img & output img network
        self.input_net = nn.Sequential()
        self.output_net = nn.Sequential()
        
        self.input_net.add_module('input_net_shape0' , PrintShape('Input Net 0'))
        self.output_net.add_module('output_net_shape0', PrintShape('Output Net 0'))
        for i in range(len(self.separated_imgnet_dims) - 1):
        
            self.input_net.add_module('input_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.input_net.add_module('input_net_relu' + str(i), nn.ReLU())
            self.input_net.add_module('input_net_shape' + str(i+1), PrintShape(
                'Input Net ' + str(i+1)))
            
            self.output_net.add_module('output_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.output_net.add_module('output_net_relu' + str(i), nn.ReLU())    
            self.output_net.add_module('output_net_shape' + str(i+1), PrintShape(
                'Output Net ' + str(i+1)))
        
        # --- Shared network
        self.shared_reprnet = nn.Sequential()
        self.shared_reprnet.add_module('shared_reprnet_shape0', PrintShape('Shared Net 0'))
        for i in range(len(self.shared_repr_dims) - 1):  
            self.shared_reprnet.add_module('shared_reprnet_lin' + str(i), nn.Linear(
                    self.shared_repr_dims[i], self.shared_repr_dims[i+1]))
            if i != len(self.shared_repr_dims) - 2:
                self.shared_reprnet.add_module('shared_reprnet_relu' + str(i), nn.ReLU())
            self.shared_reprnet.add_module('shared_reprnet_shape' + str(i+1), PrintShape(
                    'Shared Net ' + str(i+1)))
#        self.shared_reprnet.add_module('sigmoid', nn.Sigmoid())
#        self.shared_reprnet.add_module('masked_sigmoid', MaskedSigmoid())
        
        self.weight_init()
    
    def weight_init(self):
        """
        Weight initialiser.
        """
        initializer = globals()[self.opt['weight_init']]

        for block in self._modules:
            b = self._modules[block]
            if isinstance(b, nn.Sequential):
                for m in b:
                    initializer(m)
            else:
                initializer(b)
                
    def forward(self, img1, img2):
        img1_interrepr = self.shared_imgnet(img1)
        img1_repr = self.input_net(img1_interrepr)
        
        img2_interrepr = self.shared_imgnet(img2)
        img2_repr = self.output_net(img2_interrepr)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        out = self.shared_reprnet(concat_repr)
        return out




class APNet_Mlp_2Heads_1Body_1Tail(nn.Module):
    """
    Action proposal network with Linear layers.
    
    Input: latent sample from a VAE of dim latent_dim.
    Output: input_uv, input_h, output_uv
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.separated_imgnet_dims = opt['separated_imgnet_dims'] # [128, 64, 32]
        self.shared_reprnet_dims = opt['shared_reprnet_dims'] # [32*2, 16, 8, 5]
        
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        # --- Separate input img & output img network
        self.input_net = nn.Sequential()
        self.output_net = nn.Sequential()
        
        self.input_net.add_module('input_net_shape0' , PrintShape('Input 0'))
        self.output_net.add_module('output_net_shape0', PrintShape('Output 0'))
        for i in range(len(self.separated_imgnet_dims) - 1):
        
            self.input_net.add_module('input_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.input_net.add_module('input_net_relu' + str(i), nn.ReLU())
            self.input_net.add_module('input_net_shape' + str(i+1), PrintShape(
                'Input ' + str(i+1)))
            
            self.output_net.add_module('output_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.output_net.add_module('output_net_relu' + str(i), nn.ReLU())    
            self.output_net.add_module('output_net_shape' + str(i+1), PrintShape(
                'Output ' + str(i+1)))
        
        # --- Shared network
        self.shared_reprnet = nn.Sequential()
        self.shared_reprnet.add_module('shared_net_shape0', PrintShape('Shared Net 0'))
        for i in range(len(self.shared_reprnet_dims) - 1):  
            self.shared_reprnet.add_module('shared_net_lin' + str(i), nn.Linear(
                    self.shared_reprnet_dims[i], self.shared_reprnet_dims[i+1]))            
            if i != len(self.shared_reprnet_dims) - 2:
                self.shared_reprnet.add_module('shared_reprnet_relu' + str(i), nn.ReLU())
            self.shared_reprnet.add_module('shared_reprnet_shape' + str(i+1), PrintShape(
                    'Shared Net ' + str(i+1)))

        self.weight_init()
         
        
    def weight_init(self):
        """
        Weight initialiser.
        """
        initializer = globals()[self.opt['weight_init']]

        for block in self._modules:
            b = self._modules[block]
            if isinstance(b, nn.Sequential):
                for m in b:
                    initializer(m)
            else:
                initializer(b)
                
    def forward(self, img1, img2):
        img1_repr = self.input_net(img1)
        img2_repr = self.output_net(img2)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        out = self.shared_reprnet(concat_repr)        
        return out    
    

class APNet_Mlp_2Heads_1Body_2Tails(nn.Module):
    """
    Action proposal network with Linear layers.
    
    Input: latent sample from a VAE of dim latent_dim.
    Output: input_uv, input_h, output_uv
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.separate_net_dims = opt['separate_net_dims'] # [128, 64, 32]
        self.shared_net = opt['shared_net'] # [32*2, 16, 8]

        self.regression_out_dim = opt['regression_out_dim']
        self.classification_head = opt['classification_head']
        self.output_fn = self.regression_output
        
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        # --- Separate input img & output img network
        self.input_net = nn.Sequential()
        self.output_net = nn.Sequential()
        
        self.input_net.add_module('input_net_shape0' , PrintShape('Input 0'))
        self.output_net.add_module('output_net_shape0', PrintShape('Output 0'))
        for i in range(len(self.separate_net_dims) - 1):
        
            self.input_net.add_module('input_net_lin' + str(i), nn.Linear(
                    self.separate_net_dims[i], self.separate_net_dims[i+1]))    
            self.input_net.add_module('input_net_relu' + str(i), nn.ReLU())
            self.input_net.add_module('input_net_shape' + str(i+1), PrintShape(
                'Input ' + str(i+1)))
            
            self.output_net.add_module('output_net_lin' + str(i), nn.Linear(
                    self.separate_net_dims[i], self.separate_net_dims[i+1]))    
            self.output_net.add_module('output_net_relu' + str(i), nn.ReLU())    
            self.output_net.add_module('output_net_shape' + str(i+1), PrintShape(
                'Output ' + str(i+1)))
        
        # --- Shared network
        self.shared_net = nn.Sequential()
        self.shared_net.add_module('shared_net_shape0', PrintShape('Shared Net 0'))
        for i in range(len(self.shared_net_dims) - 1):  
            self.shared_net.add_module('shared_net_lin' + str(i), nn.Linear(
                    self.shared_net_dims[i], self.shared_net_dims[i+1]))
            self.shared_net.add_module('shared_net_relu' + str(i), nn.ReLU())
            self.shared_net.add_module('shared_net_shape' + str(i+1), PrintShape(
                    'Shared Net ' + str(i+1)))

        # --- Regression output
        self.regression_net = nn.Sequential(
                PrintShape('Regression Net Input'),
                nn.Linear(self.shared_net_dims[-1], self.regression_out_dim),
                nn.ReLU(),
                PrintShape('Regression Net Output'),)
        
        # --- Classification output
        if self.classification_head:
            self.classification_net = nn.Sequential(
                    PrintShape('Classification Net'),
                    nn.Linear(self.shared_net_dims[-1], opt['classification_out_dim']),
                    PrintShape('Classification Net Output'))
            self.output_fn = self.regression_classification_output
            print('Output function set to \'regression_classification_output\'')
            
        self.weight_init()
         
        
    def weight_init(self):
        """
        Weight initialiser.
        """
        initializer = globals()[self.opt['weight_init']]

        for block in self._modules:
            b = self._modules[block]
            if isinstance(b, nn.Sequential):
                for m in b:
                    initializer(m)
            else:
                initializer(b)
                
    def forward(self, img1, img2):
        img1_repr = self.input_net(img1)
        img2_repr = self.output_net(img2)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        before_out = self.shared_net(concat_repr)
        
        return self.output_fn(before_out)
        
    def regression_output(self, before_out):
        return self.regression_net(before_out)

    def regression_classification_output(self, before_out):
        return self.regression_net(before_out), self.classification_net(before_out)

# 2 versions of weight initialisation
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.Parameter)):
        m.data.fill_(0)
        print('Param_init')

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.Parameter)):
        m.data.fill_(0)
        print('Param_init')
            
            
#def create_model(opt):
#    return APNet_Mlp_1Head_2Bodies_1Tail(opt)
#    return APNet_Mlp_2Heads_1Body_1Tail(opt)

def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    size = 128
    opt = {
            'device': 'cpu',
            'shared_imgnet_dims': [128, 64, 64], # put None for 2 heads
            'separated_imgnet_dims': [64, 32, 16], # 2 heads [128, 64, 32], # 1head [64, 32, 16],
            'shared_reprnet_dims': [16*2, 16, 8, 5], # 2 heads [32*2, 16, 8, 5], # 1 head [16*2, 16, 8, 5],
#            'regression_out_dim': 4,
#            'classification_head': True, 
#            'classification_out_dim': 1, 

            'dropout': 0.2,
            'weight_init': 'normal_init',            
            }

    net = create_model(opt)
    img1 = torch.autograd.Variable(torch.FloatTensor(5, size).uniform_(-1,1))
    img2 = torch.autograd.Variable(torch.FloatTensor(5, size).uniform_(-1,1))
    print('\n * ---')
    out = net(img1, img2)
    print('\n * ---')
