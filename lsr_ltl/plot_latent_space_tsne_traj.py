import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pickle
import os, os.path
import tkinter as tk
from tkinter import messagebox
import pickle
from unity_stacking_utils import get_class_from_filename, get_actions_from_classes

import networkx as nx

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



# open a file, where you stored the pickled data
file = open('./examples/nodes_indexes1.pkl', 'rb')

# dump information to that file
l1 = pickle.load(file)

# close the file
file.close()


# open a file, where you stored the pickled data
file = open('./examples/nodes_indexes2.pkl', 'rb')

# dump information to that file
l2 = pickle.load(file)

# close the file
file.close()


config_file="VAE_push_v1"
graph_name=config_file  +"_graph"


G=nx.read_gpickle("./graphs/VAE_push_v1_graph.pkl")

print("Number of connected components:",nx.number_connected_components(G))

# Generate connected components.
print("Len of each connected component:",[len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])


# exit(0)



Z = [] # list of encoded images
C = [] # list of classes of the corrispondent z
LIST_OF_NODES = list(G.nodes)


for l in LIST_OF_NODES:
    z_pos=G.nodes[l]['pos']
    Z.append(list(z_pos))
    _class = G.nodes[l]['_class']
    C.append(_class)


all = Z

Z_traj1 = []
for l in l1:
    z_pos=G.nodes[l]['pos']
    Z_traj1.append(list(z_pos))

    all.append(list(z_pos))

Z_traj2 = []
for l in l2:
    z_pos=G.nodes[l]['pos']
    Z_traj2.append(list(z_pos))

    all.append(list(z_pos))


# print(l1)
# exit(0)

classes = [] # list of classes with no replicas

for c in C:
    found = 0
    for t in classes:
        if ((c==t).all()):
            found = 1


    if not found:
        classes.append(c)

print("number of different clases present in the dataset", len(classes))


ind = [] # list of integers that encode the class of the i-th element in Z


for c1 in C: # given a point in the latent space, I consider the class associated to it
    found = 0
    for i,c2 in enumerate(classes): #looking for the position in classes where c1 is
        if (c2==c1).all():
            ind.append(i)
            found = 1
            break
    if not found:
        print("not found",c1)



# Discern between 2 and 3 boxes configs
het = []
for c1 in C:
    if np.count_nonzero(c1)==2:
        het.append(0)
    else:
        het.append(360)

# print(len(ind))
# print(len(C))
# print(len(classes))
#
#
# print(ind[576])
# print(C[576])
# print(classes[ind[576]])

N1 = len(Z_traj1)
N2 = len(Z_traj2)
N = len(Z)-N1-N2




#
# Z = np.array(Z)
# Z_traj1 = np.array(Z_traj1)
# Z_traj2 = np.array(Z_traj2)

print(N,N1,N2)



all = np.array(all)

# print(Z.shape)
# print(Z_traj1.shape)
# print(Z_traj1)
print(all.shape)
# exit(0)



ind = np.array(ind)



# #### Standardizing the Data to have mean 0 and a standard deviation of 1
# def standardize_data(X):
#
#     '''
#     This function standardize an array, its substracts mean value,
#     and then divide the standard deviation.
#
#     param 1: array
#     return: standardized array
#     '''
#     rows, columns = X.shape
#
#     standardizedArray = np.zeros(shape=(rows, columns))
#     tempArray = np.zeros(rows)
#
#     for column in range(columns):
#
#         mean = np.mean(X[:,column])
#         std = np.std(X[:,column])
#         tempArray = np.empty(0)
#
#         for element in X[:,column]:
#
#             tempArray = np.append(tempArray, ((element - mean) / std))
#
#         standardizedArray[:,column] = tempArray
#
#     return standardizedArray



# Z = standardize_data(Z)


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# %matplotlib inline

import seaborn as sns


from mpl_toolkits.mplot3d import Axes3D


# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})




# Utility function to visualize the outputs of PCA and t-SNE

def fashion_scatter(x, colors,het,n,t,t2,tra):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # print(palette[colors.astype(np.int)])
    # create a scatter plot.
    f = plt.figure()
    if n==3:
        if tra == 1:
            ax = f.gca(projection='3d')
            # sc = ax.scatter(x[:,0], x[:,1], x[:,2],  marker='o',c=palette[colors.astype(np.int)])
            sc = ax.scatter(x[:,0], x[:,1], x[:,2], lw=0, s=40, marker='o',c=palette[het])
            ax.plot(t[:,0], t[:,1], t[:,2],  's-', label='3 boxes trajectory')
            ax.plot(t2[:,0], t2[:,1], t2[:,2],  'D-',color=palette[0], label='2 boxes trajectory')
            ax.set_xlabel("first component")
            ax.set_ylabel("second component")
            ax.set_zlabel("third component")
            ax.legend()
        else:
            ax = f.add_subplot(111, projection='3d')
            sc = ax.scatter(x[:,0], x[:,1], x[:,2], lw=0, s=40, marker='o',c=palette[colors.astype(np.int)])
            ax.set_xlabel("first component")
            ax.set_ylabel("second component")
            ax.set_zlabel("third component")

    if n ==2:
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, marker='o',c=palette[colors.astype(np.int)])
        # ax.plot(x[:,0], x[:,1],  '.',color=palette[302])
        # ax.plot(t[:,0], t[:,1], 's-', label='3 boxes trajectory')
        # ax.plot(t2[:,0], t2[:,1], 'D-',color=palette[0], label='2 boxes trajectory')
        #
        # ax.legend()


    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    ax.axis('tight')
    plt.show()
    # add the labels for each digit corresponding to the label
    txts = []

    # for i in range(num_classes):
    #
    #     # Position of each label at median of data points.
    #
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)


# Subset first 20k data points to visualize
x_subset = Z
y_subset = ind

# print( np.unique(y_subset))




from sklearn.decomposition import PCA

# time_start = time.time()
#
# pca = PCA(n_components=4)
# pca_result = pca.fit_transform(x_subset)
#
# print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
#
#
#
#
# pca_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])
#
# pca_df['pca1'] = pca_result[:,0]
# pca_df['pca2'] = pca_result[:,1]
# pca_df['pca3'] = pca_result[:,2]
# pca_df['pca4'] = pca_result[:,3]
#
# print( 'Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
#
#
#
#
#
# top_three_comp = pca_df[['pca1','pca2','pca3']] # taking first and second and third principal component

# fashion_scatter(top_three_comp.values,y_subset,3) # Visualizing the PCA output





from sklearn.manifold import TSNE
import time
# time_start = time.time()
#
# fashion_tsne = TSNE(random_state=RS).fit_transform(x_subset)
#
# print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#
#
#
# fashion_scatter(fashion_tsne, y_subset)



# time_start = time.time()
#
# pca_50 = PCA(n_components=50)
# pca_result_50 = pca_50.fit_transform(x_subset)
# # pca_result_50_traj = pca_50.fit_transform(Z_traj1)
#
# print ('PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start))
#
# print ('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))






import time
time_start = time.time()

n = 3

RS = 22
all_tsne = TSNE(n_components=n,random_state=RS,n_iter = 1000,perplexity = 50).fit_transform(all)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


fashion_pca_tsne = all_tsne[:N]
fashion_pca_tsne_traj1 = all_tsne[N:N+N1]
fashion_pca_tsne_traj2 = all_tsne[N+N1:N+N1+N2]


fashion_scatter(fashion_pca_tsne, y_subset,het,n,fashion_pca_tsne_traj1,fashion_pca_tsne_traj2,1)
fashion_scatter(fashion_pca_tsne, y_subset,het,n,fashion_pca_tsne_traj1,fashion_pca_tsne_traj2,0)
