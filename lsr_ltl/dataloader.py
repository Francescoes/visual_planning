#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:15:30 2019

@author: petrapoklukar


TODO:
     - preprocess_toy_data?
     - remove GenericTensorDataset?
     - remove binarise, binarise_threshold, grayscale, anomality from the datasetclasses
=======

"""

from __future__ import print_function
import torch
import torch.utils.data as data
import random
import sys
import pickle
import torchvision.transforms as transforms

def preprocess_triplet_data_seed(filename,seed):
    with open('datasets/'+filename, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list = pickle.load(f)
        else:
            data_list = pickle.load(f, encoding='latin1')
        #data_list = pickle.load(f)

    random.seed(int(seed))

    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    train_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    train_data))
    test_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    test_data))
    with open('datasets/train_'+filename[:-4] + "_seed_" + str(seed) + ".pkl", 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename[:-4] + "_seed_" + str(seed) + ".pkl", 'wb') as f:
        pickle.dump(test_data1, f)



def preprocess_triplet_data(filename):
    with open('datasets/'+filename, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list = pickle.load(f)
        else:
            data_list = pickle.load(f, encoding='latin1')
        #data_list = pickle.load(f)

    random.seed(2610)

    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    train_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    train_data))
    test_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    test_data))
    with open('datasets/train_'+filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename, 'wb') as f:
        pickle.dump(test_data1, f)


def preprocess_triplet_data_for_apn(filename):
    with open('datasets/'+filename, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list = pickle.load(f)
        else:
            data_list = pickle.load(f, encoding='latin1')
        #data_list = pickle.load(f)

    action_data_list = []
    for pair in data_list:
        if pair[2] == 1:
            action_data_list.append(
                (pair[0], pair[1], pair[2], pair[3][0], pair[3][1], pair[3][2]))
    random.seed(2610)
    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    train_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    train_data))
    test_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    test_data))
    with open('datasets/train_'+filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename, 'wb') as f:
        pickle.dump(test_data1, f)


def preprocess_two_triplet_datasets(filename1, filename2, common_filename):
    with open('datasets/' + filename1, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list1 = pickle.load(f)
        else:
            data_list1 = pickle.load(f, encoding='latin1')

    with open('datasets/' + filename2, 'rb') as g:
        if sys.version_info[0] < 3:
            data_list2 = pickle.load(g)
        else:
            data_list2 = pickle.load(g, encoding='latin1')

    data_list = data_list1 + data_list2
    random.seed(2610)
    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    train_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    train_data))
    test_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    test_data))
    with open('datasets/train_' + common_filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_' + common_filename, 'wb') as f:
        pickle.dump(test_data1, f)


def preprocess_triplet_data_with_augmentation(filename, flips,
                                              rotations, colorjitter,
                                              colorjitter_values=None):
    with open('datasets/'+filename, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list = pickle.load(f)
        else:
            data_list = pickle.load(f, encoding='latin1')
        #data_list = pickle.load(f)
    output_sufix = '_f{0}_r{1}_cj{2}'.format(int(flips), int(rotations), int(colorjitter))
    random.seed(2610)
    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    transforms_list = [transforms.ToPILImage()]

    if flips:
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomVerticalFlip())
    if rotations:
        transforms_list.append(transforms.RandomRotation(10))
    if colorjitter:
        b, c, s, h = colorjitter_values
        output_sufix += 'b{0}c{1}s{2}h{3}'.format(
            str(b).replace('.', 'p'), str(c).replace('.', 'p'),
            str(s).replace('.', 'p'), str(h).replace('.', 'p'))
        transforms_list.append(transforms.ColorJitter(brightness=b, #0.3,
                               contrast=c, #0.5,
                               saturation=s, #0.5,
                               hue=h))#0),
    transforms_list.append(transforms.ToTensor())
    transform = transforms.Compose(transforms_list)
    output_sufix += '.pkl'

    train_data1 = list(map(lambda t: (
        transform(torch.tensor(t[0]/255.).float().permute(2, 0, 1)),
        transform(torch.tensor(t[1]/255).float().permute(2, 0, 1)),
        torch.tensor(t[2]).float()),
                    train_data))
    test_data1 = list(map(lambda t: (
        transform(torch.tensor(t[0]/255.).float().permute(2, 0, 1)),
        transform(torch.tensor(t[1]/255).float().permute(2, 0, 1)),
        torch.tensor(t[2]).float()),
                    test_data))

    output_filename = filename.split('.')[0] + output_sufix
    with open('datasets/train_' + output_filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_' + output_filename, 'wb') as f:
        pickle.dump(test_data1, f)


# ----------------------- #
# --- Custom Datasets --- #
# ----------------------- #
class TripletTensorDataset(data.Dataset):
    def __init__(self, dataset_name, split):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split

        try:
            if split == 'test':
                with open('datasets/test_'+self.dataset_name+'.pkl', 'rb') as f:
                    self.data = pickle.load(f)
            else:
                with open('datasets/train_'+self.dataset_name+'.pkl', 'rb') as f:
                    self.data = pickle.load(f)

        except:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

    def __getitem__(self, index):
        img1, img2, action = self.data[index]
        return img1, img2, action

    def __len__(self):
        return len(self.data)


class TripletTensorDatasetClassesAct(data.Dataset):
    def __init__(self, dataset_name):

        self.dataset_name =  dataset_name
        self.name = self.dataset_name + '_'

        with open("datasets/"+self.dataset_name+'.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        img1, img2, action, a_lambda, class1, class2 = self.data[index]
        return img1, img2, action, a_lambda, class1, class2

    def __len__(self):
        return len(self.data)



class APNDataset(data.Dataset):
    def __init__(self, task_name, dataset_name, split, random_seed, dtype,
                 img_size):
        self.task_name = task_name
        self.dataset_name =  dataset_name
        self.name = dataset_name + '_' + split
        self.split = split.lower()
        self.random_seed = random_seed
        self.dtype = dtype
        self.img_size = img_size

        # Stacking data
        if self.task_name == 'push':
            path = 'action_data/{0}/{1}_{2}_seed{3}.pkl'.format(
                    self.dataset_name, self.dtype, self.split, self.random_seed)

            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.data = pickle_data['data']
                self.min, self.max = pickle_data['min'], pickle_data['max']


        # Shirt data
        if self.task_name == 'shirt_folding':
            path = './datasets/action_data/{0}/{1}_normalised_{2}_seed{3}.pkl'.format(
                    self.dataset_name, self.dtype, self.split, self.random_seed)

            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.data = pickle_data['data']
                self.min, self.max = pickle_data['min'], pickle_data['max']

    def __getitem__(self, index):
        img1, img2, coords = self.data[index]
        return img1, img2, coords

    def __len__(self):
        return len(self.data)


class LBNDataset(data.Dataset):
    def __init__(self, task_name, dataset_name, split, random_seed, dtype,
                 img_size):
        self.task_name = task_name
        self.dataset_name =  dataset_name
        self.name = dataset_name + '_' + split
        self.split = split.lower()
        self.random_seed = random_seed
        self.dtype = dtype
        self.img_size = img_size

        # Stacking data
        if self.task_name == 'push':
            path = 'LTL_data/{0}/{1}_{2}_seed{3}.pkl'.format(
                    self.dataset_name, self.dtype, self.split, self.random_seed)

            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.data = pickle_data['data']
                self.min, self.max = pickle_data['min'], pickle_data['max']



    def __getitem__(self, index):
        img1, img2, coords = self.data[index]
        return img1, img2, coords

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Test Triplet Dataset
    if False:
        batch_size = 1
        dataset = TripletTensorDataset('toy_data', 'train')

        dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

        ones = 0
        zeros = 0
        for b in dataloader:
            img1, img2, action = b
            if action == 1:
                ones += 1
            else:
                zeros +=1

        plt.figure(2)
        fig = plt.imshow(img2[0].permute(1, 2, 0))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        print(action[0])
        plt.show()
