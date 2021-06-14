"""
Creates the pkl dataset given raw data into the format: [img1, img2, a, lambda, class1, class2]'
a = binary classification if action took place
lambda = detail about the action (dependedn on task)
"""


from lib3 import *

import pickle
import math
import random
import numpy as np
import random
import sys
import torch
import itertools



#########
# Functions
#########

arr = [0,1,2]
r = 2
positions = [x for x in set(itertools.permutations(arr))]
print(positions)



###########
# Parameters
width = 800 #w captured image width
height = 800 #h captured image height

dim = (width, height) # image dimension





# pkl file name
pkl_filename = "push_v1"

# Lists
pkl_list_action=[] # list for action pairs
pkl_list_noaction=[] # list for no action pairs
###########




###########
# Simulation


# Create the environment
env = PandaPushEnv(1,w=width,h=height)
K = 7
N_action =  504*K
# 504 is the number of possible dfferent configurations we can have for 3 boxes
# in a 3x3 grid

save_img = 1 # 1 to save images
# push length
eps = .04

# Initialization - random choice of boxes position (but 3 different cells)
objectid1Pos = np.array([random.choice(sub_x)+random.uniform(-cell_size/8.,cell_size/8.),random.choice(sub_y)+random.uniform(-cell_size/8.,cell_size/8.), 0.06],dtype=object)

# print('true pos ', objectid1Pos)
while True:
    objectid2Pos = np.array([random.choice(sub_x)+random.uniform(-cell_size/8.,cell_size/8.),random.choice(sub_y)+random.uniform(-cell_size/8.,cell_size/8.), 0.06],dtype=object)
    if not(objectid2Pos==objectid1Pos).all():
        break
while True:
    objectid3Pos = np.array([random.choice(sub_x)+random.uniform(-cell_size/8.,cell_size/8.),random.choice(sub_y)+random.uniform(-cell_size/8.,cell_size/8.), 0.06],dtype=object)
    if not(objectid3Pos==objectid2Pos).all() and not(objectid3Pos==objectid1Pos).all():
        break




# Load robot and object in the chosen pose
env.reset(objectid1Pos,objectid2Pos,objectid3Pos)




# input('press for plotting...')

# list of all possible classes
all = []

num1 = 3
num2 = 3

for i in range(9):
    free1 = list(range(9))
    free1.remove(i)
    for j in free1:
        free2 = list(range(9))
        free2.remove(i)
        free2.remove(j)
        for k in free2:
            class_img = np.zeros((num1*num2,1))

            class_img[i] = 1 # blue in position i
            class_img[j] = 2 # red in position j
            class_img[k] = 3 # gree in position k
            all.append(class_img.reshape((num1,num2)))


print("len",len(all))

for k in range(K):
    for i,elem in enumerate(all):


        x1,y1 = np.where(elem==1)
        x2,y2 = np.where(elem==2)
        x3,y3 = np.where(elem==3)
        objectid1Pos = np.array([sub_x[x1]+random.uniform(-cell_size/8.,cell_size/8.),sub_y[y1]+random.uniform(-cell_size/8.,cell_size/8.), 0.06],dtype=object)
        objectid2Pos = np.array([sub_x[x2]+random.uniform(-cell_size/8.,cell_size/8.),sub_y[y2]+random.uniform(-cell_size/8.,cell_size/8.), 0.06],dtype=object)
        objectid3Pos = np.array([sub_x[x3]+random.uniform(-cell_size/8.,cell_size/8.),sub_y[y3]+random.uniform(-cell_size/8.,cell_size/8.), 0.06],dtype=object)

        class1 = elem
        env.new_config(objectid1Pos,objectid2Pos,objectid3Pos)

        # capture image
        image1 = env.captureImage()


        while True:
            # push angle choice
            push_angle = random.choice([-math.pi,-math.pi/2,0,math.pi/2])

            # push_angle = -math.pi
            # print('push_angle', push_angle)

            # cube to push choice
            ob_id = random.choice([1,2,3])
            # print('ob_id',ob_id)
            # find the position in the grid
            idx,idy = np.where(class1==ob_id)

            # check if the choice is acceptable, and if it is find the points to
            # create the trajectory the end effector should track
            if (push_angle==-math.pi/2) and (idy>0):
                if (class1[idx,idy-1]==0):
                    idx2 = idx
                    idy2 = idy-1
                    break

            if (push_angle==0) and (idx<num1-1):
                if (class1[idx+1,idy]==0) :
                    idx2 = idx+1
                    idy2 = idy
                    break

            if (push_angle==math.pi/2) and (idy<num2-1):
                if (class1[idx,idy+1]==0) :
                    idx2 = idx
                    idy2 = idy+1
                    break

            if (push_angle==-math.pi) and (idx>0):
                if (class1[idx-1,idy]==0) :
                    idx2 = idx-1
                    idy2 = idy
                    break


        # action dataset
        A = [idy[0],idx[0],idy2[0],idx2[0]]

        env.action_fun(ob_id,idx2[0],idy2[0])

        # capture image
        image2 = env.captureImage()
        # capture class
        class2 = env.getClass()
        # print(class2)

        # print(A)
        # input('press for plotting...')


        pkl_list_action.append((image1, image2, 1, A, class1, class2))

        # Save images
        if save_img:
            cv2.imwrite("Img_action/image_"+ str(k*504+i) + "_0.png",image1)
            cv2.imwrite("Img_action/image_"+ str(k*504+i) + "_1.png",image2)



###### Find no action elements
# N_noaction = int(0.5*N) # number of No-action pairs
N_noaction = 1500



for i in range(0,N_noaction):

    print('Number of noaction pairs found until now: ',i)
    image1 = pkl_list_action[-i][0]

    # find the image corrisponding to the same configuration of the pkl_list_action[i]
    # but with the boxes centered in the cells
    class_i = pkl_list_action[-i][4]
    env.noaction_fun(class_i)

    # capture image
    image0 = env.captureImage()



    pkl_list_noaction.append((image0, image1, 0, -np.ones(np.size(A)), class_i, class_i))

    # Save images
    if save_img:
        cv2.imwrite("Img_noaction/image_"+ str(i) + "_0.png",image0)
        cv2.imwrite("Img_noaction/image_"+ str(i) + "_1.png",image1)



# delete the envronment
env.close()

# print info
print('\nNumber of action pairs: ',N_action)
print('Number of no-action pairs: : ',N_noaction)




# merge the two datasets
pkl_list = pkl_list_action + pkl_list_noaction


# number of total elements
N_all = N_action + N_noaction

print('Total number of pairs: ', N_all)
print('Percentage action pairs: ', N_action/N_all*100)
print('Percentage no-action pairs: ', N_noaction/N_all*100)


# N_test = int(0.25*N_all) # number of test saples 25% total number
# N_train = N_all - N_test # number of train sample

# print('Number of test pairs: ', N_test)
# print('Number of train pairs: ', N_train)


# Write pkl files
# path = "../lsr_ltl/lsr_ltl_code/datasets/new/"
path = "./datasets/"

with open(path + pkl_filename + "3.pkl", 'wb') as f:
    pickle.dump(pkl_list, f)
