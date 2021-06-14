#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:39:12 2020

@author: petrapoklukar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Downsample, self).__init__()
        padding = int((kernel_size-1)/2)
        self.downsample_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=2, padding=padding)

    def forward(self, feat):             
        return self.downsample_layer(feat)

    
    
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


class EncoderAPNet_ConvNetMlp(nn.Module):
    """
    Encoder Action proposal network with Linear layers.
    
    Input: latent sample from a VAE of dim latent_dim.
    Output: input_uv, input_h, output_uv
    """
    def __init__(self, opt, trained_params=None):
        super().__init__()
        self.opt = opt
        self.latent_dim = opt['latent_dim']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.apn_dropout = opt['apn_dropout']
        self.encoder_dropout = opt['encoder_dropout']
        
        self.conv1_out_channels = opt['conv1_out_channels']
        self.out_channels = opt['conv1_out_channels']
        self.kernel_size = opt['kernel_size']
        self.num_scale_blocks = opt['num_scale_blocks']
        self.fc_dim = opt['fc_dim']
        self.image_size = opt['image_size']
        self.input_channels = opt['input_channels']

        # After training parameters for getting the right coords
        if trained_params is not None:
            self.data_max = trained_params['data_max']
            self.data_min = trained_params['data_min']
            self.norm_mean = trained_params['norm_mean']
            self.norm_std = trained_params['norm_std']
                
        #--- Encoder network
        self.enc_conv = nn.Sequential()
        
        self.enc_conv.add_module('enc_conv0', nn.Conv2d(
                self.input_channels, self.out_channels, self.kernel_size, 
                stride=1, padding=int((self.kernel_size-1)/2)))
        self.enc_conv.add_module('P_enc_conv0', PrintShape('Output of enc_conv0'))
        for d in range(self.num_scale_blocks):
            if d != self.num_scale_blocks - 1:
                in_channels = self.out_channels
                self.out_channels *= 2
                self.enc_conv.add_module('enc_downscale' + str(d), Downsample(
                        in_channels, self.out_channels, self.kernel_size))
                self.enc_conv.add_module('P_enc_adownscale'+str(d), PrintShape('Output of enc_downscale'))
        
        self.enc_conv.add_module('enc_avgpool', nn.AvgPool2d(3))
        self.enc_conv.add_module('P_enc_apool', PrintShape('Output of enc_avgpool'))
        self.enc_conv.add_module('enc_flatten', ConvToLin())
        self.enc_conv.add_module('P_enc_bfcs', PrintShape('Output of enc_flatten'))
        self.enc_conv.add_module('enc_lin', nn.Linear(self.fc_dim, self.latent_dim))
        self.enc_conv.add_module('P_enc_lin', PrintShape('Output of to end_lin'))
        

        self.shared_imgnet_dims = opt['shared_imgnet_dims'] # [128, 64],
        self.separated_imgnet_dims = opt['separated_imgnet_dims'] # [64, 32, 16]
        self.shared_repr_dims = opt['shared_reprnet_dims'] # [16*2, 16, 8, 5]
            
      
        # --- Shared input img & output img network
        self.shared_imgnet = nn.Sequential()
        self.shared_imgnet.add_module('shared_imgnet_shape0', PrintShape('Shared ImgNet Input 0'))
        for i in range(len(self.shared_imgnet_dims) - 1):
        
            self.shared_imgnet.add_module('shared_imgnet_lin' + str(i), nn.Linear(
                    self.shared_imgnet_dims[i], self.shared_imgnet_dims[i+1]))  
            self.shared_imgnet.add_module('shared_imgnet_dropout' + str(i), nn.Dropout(
                    p=self.apn_dropout))
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
            self.input_net.add_module('input_net_dropout' + str(i), nn.Dropout(
                p=self.apn_dropout))
            self.input_net.add_module('input_net_relu' + str(i), nn.ReLU())
            self.input_net.add_module('input_net_shape' + str(i+1), PrintShape(
                'Input Net ' + str(i+1)))
            
            self.output_net.add_module('output_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1])) 
            self.output_net.add_module('output_net_dropout' + str(i), nn.Dropout(
                p=self.apn_dropout))
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
        
        img1_encoder = self.enc_conv(img1)
        img1_interrepr = self.shared_imgnet(img1_encoder)
        img1_repr = self.input_net(img1_interrepr)

        img2_encoder = self.enc_conv(img2)
        img2_interrepr = self.shared_imgnet(img2_encoder)
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
        
        
def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    size = 256
    opt = {
            'device': 'cpu',
            'image_size': size,
            'input_dim': 256*256*3,
            'input_channels': 3,
            'latent_dim': 16,
            'conv1_out_channels': 8, 
            'kernel_size': 3, 
            'num_scale_blocks': 6,
            'block_per_scale': 1,
            'depth_per_block': 2, 
            'fc_dim': 1024,    
            'encoder_dropout': 0.2,
        
            'shared_imgnet_dims': [16, 64],
            'separated_imgnet_dims': [64, 16],
            'shared_reprnet_dims': [16*2, 16, 5],     
            'apn_dropout': 0.2,
            'weight_init': 'normal_init',    
            }
    
    net = EncoderAPNet_ConvNetMlp(opt)
    img1 = torch.autograd.Variable(torch.FloatTensor(5, 3, size, size).uniform_(-1,1))
    img2 = torch.autograd.Variable(torch.FloatTensor(5, 3, size, size).uniform_(-1,1))
    print('\n * ---')
    print(img1.shape)
    out = net(img1, img2)
    print(out.shape)
    print('\n * ---')



