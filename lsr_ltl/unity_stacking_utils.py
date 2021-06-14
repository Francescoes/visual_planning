import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pickle
import os, os.path
import tkinter as tk
from tkinter import messagebox
import pickle

"""
Utility functions for box stacking example and main function for trackbar (for mask)
"""


####################### GLOBAL VARIABLES
size_gf = 9
dict_num_colors = {"pink":3, "yellow":1, "orange":2, "white":5}
# Threshold of colors in HSV space
lower_pink = np.array([0, 0, 148])
upper_pink = np.array([10, 250, 222])
lower_yellow = np.array([20, 110, 0])
upper_yellow = np.array([179, 255, 255])
lower_orange = np.array([0, 75, 156])
upper_orange = np.array([17, 255, 255])
lower_white = np.array([55, 0, 193])
upper_white = np.array([137, 93, 255])


# lower_yellow_h = np.array([15, 72, 0])
# upper_yellow_h = np.array([90, 255, 255])
# lower_orange_h = np.array([0, 67, 0])
# upper_orange_h = np.array([11, 255, 255])
# lower_white_h = np.array([0, 0, 181])
# upper_white_h = np.array([67, 25, 255])

# lower_brown_h = np.array([17, 28, 58])
# upper_brown_h = np.array([80, 61, 177])

#dict_num_colors_h = {"brown":2, "yellow":5, "orange":1, "white":3}

####################### END  GLOBAL VARIABLES


def nothing(x):
    pass



def get_class_from_filename(filename):
    """
    Returns the class of the image

    - filename: name of the file with the image
    """
    gamefield=np.zeros(size_gf, dtype=np.int)
    idx_start = 4
    idx_end = idx_start+size_gf
    gamefield_str = filename.split("_")[idx_start:idx_end]

    for i in range(len(gamefield_str)):
        gamefield[i] = int(gamefield_str[i])

    return gamefield

def get_actions_from_classes(class1, class2):
    """
    Returns the action between consecutive classes

    - class1, class2: consecutive classes
    """

    #check gamefield
    pick_action=[-1,-1]
    place_action=[-1,-1]
    for i in range(len(class1)):
        if not class1[i] == class2[i]:
                if class1[i] >0 and class2[i] ==0:
                    pick_action[1]=int(i/3)
                    pick_action[0]=int(i%3)
                if class1[i] ==0 and class2[i] >0:
                    place_action[1]=int(i/3)
                    place_action[0]=int(i%3)


    return [pick_action, place_action]

def get_col_row_from_centroid(c):
    """
    Returns column and row indices (in the 3x3 grid) from box centroid

    - c: box centroid
    """
    cX = c[0]
    cY = c[1]
    if cX < 75:
        col = 0
    elif cX < 150:
        col = 1
    else:
        col = 2

    if cY < 100:
        row = 0
    elif cY < 155:
        row = 1
    else:
        row = 2

    return row,col

def get_box_center_from_x_y(x, y):
    """
    Returns the center of the box from grid indices
    it assumes top left corner = (0,0), x goes positively towards down, y towards right
    """
    cx_vec = [55,115,185]#according to image coordinates (x positive towards right)
    cy_vec = [87,140,190]#according to image coordinates (y positive towards down)

    cx = cx_vec[y]
    if x == 2:
        cy = 195
    else:
        cy = cy_vec[x]

    return (cx,cy)

def get_centroids_colors(hsv, SHOW_IMAGES = False, num_blocks = 3):
    """
    Returns the centroids cordinates and colors
    """
    # preparing the mask to overlay
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    masks = {"pink":mask_pink, "yellow":mask_yellow, "orange":mask_orange}

    if num_blocks == 4:
        masks["white"] = mask_white


    centers = dict()
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)
    kernel_dil = np.ones((10,10), np.uint8)
    c = 0
    cen_list_x = []
    cen_list_y = []
    for key in masks:
        mask = masks[key]
        mask[-20:-1, 0:80] = 0

        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel_dil, iterations=1)
        #cv2.imshow(key, mask)
        # calculate moments of binary image
        M = cv2.moments(mask)


        if int(M["m00"]) == 0:
            return []

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


        # put text and highlight the center
        cv2.circle(mask, (cX, cY), 5, 0, -1)
        cen_list_x.append(cX)
        cen_list_y.append(cY)
        if SHOW_IMAGES:
            cv2.imshow('mask '+str(key), mask)
            cv2.waitKey(0)
        c+=1
        centers[key] = (cX,cY)
    return centers

def get_class_from_hsv_image(hsv, num_blocks = 3):
    """
    Returns the class from the hsv image
    """
    centers = get_centroids_colors(hsv, num_blocks = num_blocks)
    size = int(np.sqrt(size_gf))
    gamefield=np.zeros((size,size), dtype=np.int)
    for key in centers:
        num_color = dict_num_colors[key]
        row, col = get_col_row_from_centroid(centers[key])
        gamefield[row][col] = num_color

    return np.reshape(gamefield, size_gf)


def main(args):
    TRACKBAR = False
    SHOW_IMAGES = True

    cen_list_x = []
    cen_list_y = []
    #base_main_folder="4blocks"
    base_main_folder = "Horror"
    num_blocks = 4
    print("num blocks:"+str(num_blocks))

    #base_folder = base_main_folder+"/vae"
    base_folder = base_main_folder+"/graph"

    #check folder
    path, dirs, files = next(os.walk(base_folder))
    num_images = len(files)
    width = 256
    height = 256
    dim = (width, height)

    if len(files)>0:
        numdigits = len(files[0].split("_")[0]) #get the initial string up to the first _


    print("found " +str(num_images))

    n_samples = num_images #1000
    i = 0
    for file in files:
        #check if it is a no action file
        if file[-5]=="-":
            img_org = cv2.imread(base_folder+"/" + file)
            #crop image
            img_org=img_org[0:480,160:640]

            img_org = cv2.resize(img_org, dim, interpolation = cv2.INTER_AREA)

            hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
            # width_or, height_or = np.size(img_org)

            if TRACKBAR:
                """
                trackbar
                """
                filename = "masktr"
                cv2.namedWindow(filename,1)

                #set trackbar
                hh = 'hue high'
                hl = 'hue low'
                sh = 'saturation high'
                sl = 'saturation low'
                vh = 'value high'
                vl = 'value low'
                mode = 'mode'

                #set ranges
                default_vec_low = lower_pink
                default_vec_up = upper_pink
                cv2.createTrackbar(hh, filename, default_vec_up[0],179, nothing)
                cv2.createTrackbar(hl, filename, default_vec_low[0],179, nothing)
                cv2.createTrackbar(sh, filename, default_vec_up[1],255, nothing)
                cv2.createTrackbar(sl, filename, default_vec_low[1],255, nothing)
                cv2.createTrackbar(vh, filename, default_vec_up[2],255, nothing)
                cv2.createTrackbar(vl, filename, default_vec_low[2],255, nothing)
                cv2.createTrackbar(mode, filename, 0,3, nothing)

                thv= 'th1'
                cv2.createTrackbar(thv, filename, 127,255, nothing)
                hsv_img = hsv

                while True:
                    hul= cv2.getTrackbarPos(hl,filename)
                    huh= cv2.getTrackbarPos(hh,filename)
                    sal= cv2.getTrackbarPos(sl,filename)
                    sah= cv2.getTrackbarPos(sh,filename)
                    val= cv2.getTrackbarPos(vl,filename)
                    vah= cv2.getTrackbarPos(vh,filename)
                    thva= cv2.getTrackbarPos(thv,filename)

                    modev= cv2.getTrackbarPos(mode,filename)

                    hsvl = np.array([hul, sal, val], np.uint8)
                    hsvh = np.array([huh, sah, vah], np.uint8)

                    mask = cv2.inRange(hsv_img, hsvl, hsvh)

                    if modev ==0:
                        #show mask only
                        cv2.imshow(filename,mask)
                    elif modev ==1:
                        #show white-masked color img
                        cv2.imshow(filename,res)
                    elif modev ==2:
                        #show white-masked binary img with threshold
                        cv2.imshow(filename,threshold)
                    else:
                        #white-masked grayscale img with threshold
                        cv2.imshow(filename,res2)

                    #press 'Esc' to close the window
                    ch = cv2.waitKey(5)
                    if ch== 27:
                        break
                cv2.destroyAllWindows()

                """
                end trackbar
                """


            class_from_image = get_class_from_hsv_image(hsv,num_blocks = num_blocks)
            class_gt = get_class_from_filename(file)
            if not np.array_equal(class_gt,class_from_image):
                print("Error class:"+str(class_from_image)+" gt: "+str(class_gt))
            if SHOW_IMAGES:
                cv2.imshow('Source image', img_org)

                cv2.waitKey()

            i+= 1
            if i>n_samples:
                break


    # cv2.imshow('calcHist Demo', histImage)
    plt.figure(1)
    plt.hist(cen_list_x,bins = 100)
    plt.title("x pos")
    plt.xlabel('variable X (bin size = 5)')
    plt.ylabel('count')
    plt.figure(2)
    plt.hist(cen_list_y,bins = 100)
    plt.title("y pos")
    plt.xlabel('variable X (bin size = 5)')
    plt.ylabel('count')

    plt.show()


if __name__ == "__main__":
    main(sys.argv)
