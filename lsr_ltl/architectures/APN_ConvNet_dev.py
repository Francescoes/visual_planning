#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:30:07 2020

@author: petrapoklukar
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init    
    
class LinToConv(nn.Module):
    def __init__(self, input_dim, n_channels):
        super(LinToConv, self).__init__()
        self.n_channels = n_channels
        self.width = int(np.sqrt((input_dim / n_channels)))

    def forward(self, feat):
        feat = feat.view((feat.shape[0], self.n_channels, self.width, self.width))
        return feat
    
    
class ConvToLin(nn.Module):
    def __init__(self): 
        super(ConvToLin, self).__init__()

    def forward(self, feat):
        batch, channels, width, height = feat.shape
        feat = feat.view((batch, channels * width * height)) 
        return feat


class PrintShape(nn.Module):
    def __init__(self, message):
        super(PrintShape, self).__init__()
        self.message = message
        
    def forward(self, feat):
#        print(self.message, feat.shape)
        return feat 


class APN_ConvNet_2Heads_1Body_1Tail(nn.Module):
    """
    
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.separated_imgnet_dims = opt['separated_imgnet_dims'] # [128, 64, 32]
        self.separated_imgnet_kernels = opt['separated_imgnet_kernels']
        self.shared_reprnet_dims = opt['shared_reprnet_dims'] # [32*2, 16, 8, 5]
        
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        # --- Separate input img & output img network
        self.input_net = nn.Sequential()
        self.output_net = nn.Sequential()
        
        self.input_net.add_module('input_net_shape0' , PrintShape('Input 0'))
        self.output_net.add_module('output_net_shape0', PrintShape('Output 0'))
        
        for i in range(len(self.separated_imgnet_dims) - 1):            
            self.input_net.add_module('input_conv'+ str(i), nn.Conv2d(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1], 
                    self.separated_imgnet_kernels[i], stride=2, padding=0))
            self.input_net.add_module('input_bn' + str(i), nn.BatchNorm2d(self.separated_imgnet_dims[i+1]))
            self.input_net.add_module('input_relu' + str(i), nn.ReLU())
            self.input_net.add_module('input_P_arelu'+str(i), PrintShape('Input to conv'))
            if i > 0:
                self.input_net.add_module('input_pool'+str(i), nn.AvgPool2d(self.separated_imgnet_kernels[i]))
                self.input_net.add_module('input_P_apool'+str(i), PrintShape('Output od pool'))
            
            self.output_net.add_module('input_conv'+ str(i), nn.Conv2d(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1], 
                    self.separated_imgnet_kernels[i], stride=2, padding=0))
            self.output_net.add_module('input_P_arelu'+str(i), PrintShape('Input to conv'))
            if i > 0:
                self.output_net.add_module('input_pool'+str(i), nn.AvgPool2d(self.separated_imgnet_kernels[i]))
                self.output_net.add_module('input_P_apool'+str(i), PrintShape('Output of pool'))

        self.input_net.add_module('input_flatten', ConvToLin())
        self.output_net.add_module('output_flatten', ConvToLin())
        
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
    
    
class APN_ConvNet_1Head_2Bodies_1Tail(nn.Module):
    """
    
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.shared_imgnet_dims = opt['shared_imgnet_dims'] # [128, 64],
        self.shared_imgnet_kernels = opt['shared_imgnet_kernels']
        self.separated_imgnet_dims = opt['separated_imgnet_dims'] # [128, 64, 32]
        self.separated_imgnet_kernels = opt['separated_imgnet_kernels']
        self.shared_reprnet_dims = opt['shared_reprnet_dims'] # [32*2, 16, 8, 5]
        
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        # --- Shared input img & output img network
        self.shared_imgnet = nn.Sequential()
        self.shared_imgnet.add_module('shared_imgnet_shape0', PrintShape('Shared ImgNet Input 0'))
        for i in range(len(self.shared_imgnet_dims) - 1):
            self.shared_imgnet.add_module('shared_imgnet_conv'+ str(i), nn.Conv2d(
                    self.shared_imgnet_dims[i], self.shared_imgnet_dims[i+1], 
                    self.shared_imgnet_kernels[i], stride=2, padding=0))
            self.shared_imgnet.add_module('shared_imgnet_bn' + str(i), nn.BatchNorm2d(self.shared_imgnet_dims[i+1]))
            self.shared_imgnet.add_module('shared_imgnet_relu' + str(i), nn.ReLU())
            self.shared_imgnet.add_module('shared_imgnet_P_arelu'+str(i), PrintShape('Shared ImgNet Input to conv'))
        self.shared_imgnet.add_module('shared_imgnet_pool'+str(i), nn.AvgPool2d(2))
        self.shared_imgnet.add_module('shared_imgnet_P_apool'+str(i), PrintShape('shared_imgnet Output of pool'))
        
        # --- Separate input img & output img network
        self.input_net = nn.Sequential()
        self.output_net = nn.Sequential()
        
        self.input_net.add_module('input_net_shape0' , PrintShape('Input 0'))
        self.output_net.add_module('output_net_shape0', PrintShape('Output 0'))
        
        for i in range(len(self.separated_imgnet_dims) - 1):            
            self.input_net.add_module('input_conv'+ str(i), nn.Conv2d(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1], 
                    self.separated_imgnet_kernels[i], stride=2, padding=0))
            self.input_net.add_module('input_bn' + str(i), nn.BatchNorm2d(self.separated_imgnet_dims[i+1]))
            self.input_net.add_module('input_relu' + str(i), nn.ReLU())
            self.input_net.add_module('input_P_arelu'+str(i), PrintShape('Input to conv'))
            if i == 0:
                self.input_net.add_module('input_pool'+str(i), nn.AvgPool2d(self.separated_imgnet_kernels[i]))
                self.input_net.add_module('input_P_apool'+str(i), PrintShape('Output od pool'))
            
            self.output_net.add_module('input_conv'+ str(i), nn.Conv2d(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1], 
                    self.separated_imgnet_kernels[i], stride=2, padding=0))
            self.output_net.add_module('input_P_arelu'+str(i), PrintShape('Input to conv'))
            if i == 0:
                self.output_net.add_module('input_pool'+str(i), nn.AvgPool2d(self.separated_imgnet_kernels[i]))
                self.output_net.add_module('input_P_apool'+str(i), PrintShape('Output of pool'))

        self.input_net.add_module('input_flatten', ConvToLin())
        self.output_net.add_module('output_flatten', ConvToLin())
        
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
        img1_interrepr = self.shared_imgnet(img1)
        img1_repr = self.input_net(img1_interrepr)
        
        img2_interrepr = self.shared_imgnet(img2)
        img2_repr = self.output_net(img2_interrepr)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        out = self.shared_reprnet(concat_repr)
        return out
  
    
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
            
            
def create_model(opt):
    return APN_ConvNet_1Head_2Bodies_1Tail(opt)


def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

      
if __name__ == '__main__':
    size = 256
    opt = {
            'device': 'cpu',
            'dropout': 0.2,
            'weight_init': 'normal_init',
            
            'shared_imgnet_dims': [3, 16, 32],  # [128, 64],
            'shared_imgnet_kernels': [5, 3], 
            'separated_imgnet_dims': [32, 96, 128], 
            'separated_imgnet_kernels': [3, 3], 
            'shared_reprnet_dims': [1024, 512, 256, 64, 5],
            'image_size': 256,

            }

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(5, 3, size, size).uniform_(-1,1))
    y = torch.autograd.Variable(torch.FloatTensor(5, 3, size, size).uniform_(-1,1))
    print('\n * ---')
    out = net(x, y)
    print('\n * ---')
    print(x.shape)
    print(out[-1].shape)
    print(out[0].shape)