from lib3 import *

import pickle
import math
import random
import numpy as np
import random
import sys
import torch


#########
# Functions
#########




###########
# Parameters
width = 800 #w captured image width
height = 800 #h captured image height

dim = (width, height) # image dimension


save_img = 1 # 1 to save images


###########

open_file = open("actions/actions_to_run1.pkl", "rb")

actions1 = pickle.load(open_file)

open_file.close()


open_file = open("actions/actions_to_run2.pkl", "rb")

actions2 = pickle.load(open_file)

open_file.close()

###########
# Simulation


# Create the environment
env = PandaPushEnv(1,w=width,h=height)


# push length
eps = .042


# Initialization - random choice of boxes position (but 3 different cells)
objectid1Pos = np.array([sub_x[0],sub_y[0], 0.1],dtype=object)# blue
objectid2Pos = np.array([sub_x[2],sub_y[0], 0.1],dtype=object)# red
objectid3Pos = np.array([sub_x[1],sub_y[2], 0.1],dtype=object)# green




# Load robot and object in the chosen pose
env.reset(objectid1Pos,objectid2Pos,objectid3Pos)

# capture current imageinput('press something...')
image1 = env.captureImage()



# Save image
if save_img:
    cv2.imwrite("Img_path/image_"+ str(0) + ".png",image1)



input('press something...')

for i in range(len(actions1)):
# for i in range(1):
    print('\n',i,'# execution')

    print(actions1[i])
    idy,idx,H,idy2,idx2 = actions1[i]

    if (idx == idx2) and (idy2 == idy-1):
        push_angle = -math.pi/2
        pos_ob = np.array([sub_x[idx],e_y[idy+1]+eps*.1,0.1],dtype=object)
        pos_ob2 = np.array([sub_x[idx2],e_y[idy2+1]-eps,0.1],dtype=object)
    elif (idx == idx2) and (idy2 == idy+1):
        push_angle = +math.pi/2
        pos_ob = np.array([sub_x[idx],e_y[idy]-eps*.1,0.1],dtype=object)
        pos_ob2 = np.array([sub_x[idx2],e_y[idy2]+eps,0.1],dtype=object)
    elif (idx2 == idx+1) and (idy2 == idy):
        push_angle = 0
        pos_ob = np.array([e_x[idx]-eps*.1,sub_y[idy],0.1],dtype=object)
        pos_ob2 = np.array([e_x[idx2]+eps,sub_y[idy2],0.1],dtype=object)
    elif (idx2 == idx-1) and (idy2 == idy):
        push_angle = -math.pi
        pos_ob = np.array([e_x[idx+1]+eps*.1,sub_y[idy],0.1],dtype=object)
        pos_ob2 = np.array([e_x[idx2+1]-eps,sub_y[idy2],0.1],dtype=object)


    a = np.concatenate((push_angle,pos_ob,pos_ob2),axis=None)
    observation, reward, done = env.step(a)



    # capture image
    image2 = env.captureImage()
    # input('press something...')





    # Save images
    if save_img:
        cv2.imwrite("Img_path/image_"+ str(i+1) + ".png",image2)

input('press something...')


position1 = observation[7:10]
orientation1 = observation[10:14]
position2 = observation[14:17]+10
orientation2 = observation[17:21]
position3 = observation[21:24]
orientation3 = observation[24:28]

env.Removebox()
# exit(0)

image1 = env.captureImage()
# input('Red box removed! Press something to continue...')

for i in range(len(actions2)):
    print('\n',i,'# execution')

    pos_goal = np.array([e_x[1],sub_y[1],0.3],dtype=object)
    env.move(pos_goal,0)

    idy,idx,H,idy2,idx2 = actions2[i]

    if (idx == idx2) and (idy2 == idy-1):
        push_angle = -math.pi/2
        pos_ob = np.array([sub_x[idx],e_y[idy+1]+eps*.1,0.1],dtype=object)
        pos_ob2 = np.array([sub_x[idx2],e_y[idy2+1]-eps,0.1],dtype=object)
    elif (idx == idx2) and (idy2 == idy+1):
        push_angle = +math.pi/2
        pos_ob = np.array([sub_x[idx],e_y[idy]-eps*.1,0.1],dtype=object)
        pos_ob2 = np.array([sub_x[idx2],e_y[idy2]+eps,0.1],dtype=object)
    elif (idx2 == idx+1) and (idy2 == idy):
        push_angle = 0
        pos_ob = np.array([e_x[idx]-eps*.1,sub_y[idy],0.1],dtype=object)
        pos_ob2 = np.array([e_x[idx2]+eps,sub_y[idy2],0.1],dtype=object)
    elif (idx2 == idx-1) and (idy2 == idy):
        push_angle = -math.pi
        pos_ob = np.array([e_x[idx+1]+eps*.1,sub_y[idy],0.1],dtype=object)
        pos_ob2 = np.array([e_x[idx2+1]-eps,sub_y[idy2],0.1],dtype=object)


    a = np.concatenate((push_angle,pos_ob,pos_ob2),axis=None)
    observation, reward, done = env.step(a)


    # capture image
    image2 = env.captureImage()





    # Save images
    if save_img:
        cv2.imwrite("Img_path/image_"+ str(i+1) + ".png",image2)

input('press something...')

# delete the envronment once th smulation is over
env.close()
