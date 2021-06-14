import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import glob
import time
from datetime import datetime
# import h5py
from scipy.spatial import distance

import cv2
import matplotlib.pyplot as plt
MAX_EPISODE_LEN = 20*100


offset_base = np.array([50.,50.,0.])*0

# Region reachable by the robot
min = 0.28 + offset_base[0]
max = 0.73 + offset_base[0]
min2 = -0.225 + offset_base[1]
max2 = 0.225 + offset_base[1]
cons3 = 0.05



# Offset camera object
offset = np.array([10-0.02,10,0.])
#place the camera at a desired position and orientation in the environment
view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[(max-min)/2+min,offset_base[1],-1]+offset,
        distance=5,
        yaw=90,
        pitch=-90,
        roll=0,
        upAxisIndex=2)

proj_matrix = p.computeProjectionMatrixFOV(
        fov=6.5,
        aspect=float(960) /720,
        nearVal=0.1,
        farVal=100.0)



# Grid for classes definition
cell_size = 0.1 # cube_size / cons
num1 = 3 # cells number along x
num2 = 3 # cells number along y
# print('n',num1,num2)
s = num1*num2 # number of cells

# find the center of each cell along x
e_x,step_x = np.linspace(min, max, num=num1+1,retstep = True) # e = spigolo
c_x = e_x + step_x/2 # centers
sub_x = c_x[:-1] # this would go out of the grid
# print(e_x)
# find the center of each cell along y
e_y,step_y = np.linspace(min2, max2, num=num2+1,retstep = True) # e = spigolo
c_y = e_y + step_y/2 # centers
sub_y = c_y[:-1] # this would go out of the grid





# Remove robot and target objects
# print('visual: ',p.getVisualShapeData(self.pandaUid))

pandaNumDofs = 7
ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs


#rest position of the robot
rp = [0,-0.215,0,-2.57,0,2.356,0,0.008,0.008]


class PandaPushEnv:
    def __init__(self,gui=0,w=100,h=100):
        self.step_counter = 0

        if gui:
            p.connect(p.GUI,options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        else:
            p.connect(p.DIRECT)
        #p.connect(p.GUI,options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2] + offset_base)#the initial view of the environment
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)

        p.setRealTimeSimulation(0)
        # p.setRealTimeSimulation(1)

        self.w = w
        self.h = h


        # self.logId = -1





    # compute inverse kinematics and motor control
    def motor_control(self, pos, orn):




        jointPoses = p.calculateInverseKinematics(self.pandaUid, self.pandaEndEffectorIndex, pos, orn, ll, ul, jr, rp)



        for i in range(pandaNumDofs):
            #print(jointPoses[i])
            p.setJointMotorControl2(bodyIndex=self.pandaUid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)

        p.stepSimulation()
        time.sleep(self.timeStep)

        # self.plot_traj(pos)

    # Move to goal pose
    def move(self,pos_goal,push_angle):
        pos_cur = self.observation[0:3]
        or_cur =  p.getEulerFromQuaternion(self.observation[3:7])[2]
        # print('or',or_cur)
        # print('or_cur', or_cur)
        de = pos_goal-pos_cur
        do = push_angle-or_cur
        err_pos = np.linalg.norm(de)
        err_or = np.linalg.norm(do)


        s = 0.005
        k_p = 10*s
        k_d = 1*s
        dt = 1./240. # the default timestep in pybullet is 240 Hz

        threshold_pos = 0.02
        threshold_or = 0.001
        count = 0



        while ((err_pos>threshold_pos) | (err_or>threshold_or) and count<200):
            pos_cur =self.observation[0:3]
            or_cur =  p.getEulerFromQuaternion(self.observation[3:7])[2]
            de = pos_goal-pos_cur
            do = push_angle-or_cur
            err_pos = np.linalg.norm(de)
            err_or = np.linalg.norm(do)

            pd = k_p*de + k_d*de/dt
            od = k_p*do + k_d*do/dt
            # print('Error pos', err_pos)
            # print('Error or', do)

            new_pos = pos_cur + pd
            new_or = p.getQuaternionFromEuler([-math.pi,0.,or_cur + od])
            self.motor_control(new_pos, new_or)

            self._get_state()
            count=count+1

        return self.observation[0:3],p.getEulerFromQuaternion(self.observation[3:7])[2]




    # execute action
    def step(self, action):
        #ACTIONS:
        # - push angle
        # - length trajectory

        #OBSERVATIONS:
        # - object 3Dposition
        # - end effector 3Dposition
        # - end effector orientation quaternion
        # - object orientation quaternion

        #reach safe position and change end-effector orientation
        print('Manipulator action execution')
        print('reach safe position and change end-effector orientation')

        push_angle = action[0]
        z_down = 0.2
        pos_goal = action[1:4]
        pos_goal[2] =  z_down
        pos_back = pos_goal
        c=1
        if push_angle == -math.pi:
            c = 0
        pos_cur,or_cur = self.move(pos_goal,push_angle*c)

        #reach position from which start pushing
        print('reach position from which start pushing')
        z_down = 0.01
        pos_goal[2] = z_down
        pos_back = pos_goal
        pos_cur,or_cur = self.move(pos_goal,push_angle*c)


        # pushing
        print('pushing')
        pos_goal = action[4:]
        z_down = 0.01
        pos_goal[2] = z_down
        # pos_goal[2] = pos_goal[2] + 0.02
        pos_cur,or_cur = self.move(pos_goal,push_angle*c)




        # going back
        print('going back')
        pos_goal = pos_back

        pos_cur,or_cur = self.move(pos_goal,push_angle*c)


        # # Initial position robot
        # for i in range(7):
        #     p.resetJointState(self.pandaUid,i, rp[i])
        # p.resetJointState(self.pandaUid, 9, rp[7])
        # p.resetJointState(self.pandaUid,10, rp[8])


        pos1, or1 = self.get_pose_ob(1)
        pos2, or2 = self.get_pose_ob(2)
        pos3, or3 = self.get_pose_ob(3)
        self.changeObjPoseCamera(pos1+offset,or1,pos2+offset,or2,pos3+offset,or3)


        # reset orientation
        print('reset orientation end effector')
        z_down = 0.2
        pos_goal[2] =  z_down

        pos_cur,or_cur = self.move(pos_goal,0)

        reward = 1
        done = True


        return np.array(self.observation).astype(np.float32), reward, done

    def reset(self,objectid1Pos,objectid2Pos,objectid3Pos):
        self.step_counter = 0
        self.prevPose = [0, 0, 0]
        self.prevPose1 = [0, 0, 0]
        self.hasPrevPose = 0
        self.trailDuration = 15


        p.resetSimulation()#reset the PyBullet environment
        p.setPhysicsEngineParameter(numSolverIterations=150)
        fps=120.
        self.timeStep = 1./fps
        p.setTimeStep(self.timeStep)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # disable the rendering

        p.setGravity(0,0,-9.81)

        # Load objects
        urdfRootPath = pybullet_data.getDataPath()
        # Plane
        self.planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,-.5,0.0])

        # # Texture plane
        # texture_path = './colors/white.png'
        # textureId = p.loadTexture(texture_path)
        # p.changeVisualShape(self.planeUid , -1, textureUniqueId=textureId)
        # p.changeVisualShape(self.planeUid , -1, rgbaColor=[1,1,1,1])






        # Cube 1
        gS = 1
        self.objectid1 = p.loadURDF("./urdf/cube_grid.urdf",objectid1Pos, [0,0,0,1] ,globalScaling = gS)

        # Texture Cube
        texture_path = './colors/blue.jpeg'
        textureId = p.loadTexture(texture_path)
        p.changeVisualShape(self.objectid1 , -1, textureUniqueId=textureId)
        # p.changeVisualShape(self.objectUid , -1, rgbaColor=[1,1,1,1])


        # Box camera
        self.objectCameraUid1 = p.loadURDF("./urdf/cube_grid.urdf", objectid1Pos + offset, [0,0,0,1] ,globalScaling = gS)
        p.changeVisualShape(self.objectCameraUid1, -1, textureUniqueId=textureId)



        # Cube 2
        self.objectid2 = p.loadURDF("./urdf/cube_grid.urdf",objectid2Pos,[0,0,0,1],globalScaling = gS)
        # p.changeVisualShape(self.objectid2 , -1, rgbaColor=[1,0,0,1])
        # Texture Cube
        texture_path = './colors/red.jpeg'
        textureId = p.loadTexture(texture_path)
        p.changeVisualShape(self.objectid2 , -1, textureUniqueId=textureId)

        # Box camera
        self.objectCameraUid2 = p.loadURDF("./urdf/cube_grid.urdf",objectid2Pos + offset, [0,0,0,1] ,globalScaling = gS)
        p.changeVisualShape(self.objectCameraUid2 , -1, textureUniqueId=textureId)


        # Cube 3
        self.objectid3 = p.loadURDF("./urdf/cube_grid.urdf",objectid3Pos, [0,0,0,1],globalScaling = gS)
        # p.changeVisualShape(self.objectid3 , -1, rgbaColor=[0,1,0,1])

        # Texture Cube
        texture_path = './colors/green.jpeg'
        textureId = p.loadTexture(texture_path)
        p.changeVisualShape(self.objectid3 , -1, textureUniqueId=textureId)

        # Box camera
        self.objectCameraUid3 = p.loadURDF("./urdf/cube_grid.urdf",objectid3Pos + offset, [0,0,0,1] ,globalScaling = gS)
        p.changeVisualShape(self.objectCameraUid3 , -1, textureUniqueId=textureId)


        # Table
        #tableUid = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"),basePosition=[0.5,0,-0.65],globalScaling = 15)

        # Sphere
        # sphereStartPos = [1,0,0.1]
        # sphereStartPos = cubeStartPos
        # self.sphereid = p.loadURDF(os.path.join(urdfRootPath,"sphere2red_nocol.urdf"),sphereStartPos, cubeStartOrientation,globalScaling = cell_size)





        # Franka Panda robot
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), offset_base,useFixedBase=True)
        self.pandaEndEffectorIndex = 11

        # Initial position robot
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rp[i])
        p.resetJointState(self.pandaUid, 9, rp[7])
        p.resetJointState(self.pandaUid,10, rp[8])


        # Enable the rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        # Get current state
        self._get_state()


        # Plot of the grid and the reachable region
        lineWidth = 1.5
        # plot grid
        # print(e_x)
        # Grid
        color = [0, 0, 0]
        for k in range(0,num1+1):
            p.addUserDebugLine((e_x[k], e_y[0], 0.01), (e_x[k], e_y[-1],0.01), color, lineWidth)

        for k in range(0,num2+1):
            p.addUserDebugLine((e_x[0], e_y[k], 0.01), (e_x[-1], e_y[k],0.01), color, lineWidth)


        # if self.logId==-1:
        #     self.logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video.mp4")


        # # Reachable region
        # color2 = [1, 0, 0]
        # p.addUserDebugLine((min, min2, 0.01), (max,min2,0.01), color2, lineWidth)
        # p.addUserDebugLine((min, min2, 0.01), (min,max2,0.01), color2, lineWidth)
        # p.addUserDebugLine((max, min2, 0.01), (max,max2,0.01), color2, lineWidth)
        # p.addUserDebugLine((max, max2, 0.01), (min,max2,0.01), color2, lineWidth)




    def plot_traj(self,pos):
        # plot trajectory
        ls = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)
        if (self.hasPrevPose):
            # p.addUserDebugLine(self.prevPose, pos, [0, 0, 0.3], 1, self.trailDuration)
            p.addUserDebugLine(self.prevPose1, ls[4], [1, 0, 0], 1, self.trailDuration)
        self.prevPose = pos
        self.prevPose1 = ls[4]
        self.hasPrevPose = 1



    def _get_state(self):

        state_robot_pos = p.getLinkState(self.pandaUid, 11)[0] #3D pos end-effector
        state_robot_or = p.getLinkState(self.pandaUid, 11)[1] #4D orientation end-effector
        state_object_pos1, state_object_or1 = p.getBasePositionAndOrientation(self.objectid1)
        state_object_pos2, state_object_or2 = p.getBasePositionAndOrientation(self.objectid2)
        state_object_pos3, state_object_or3 = p.getBasePositionAndOrientation(self.objectid3)
        # print('Link state',p.getLinkState(self.pandaUid, 11))
        # print('Euler', p.getEulerFromQuaternion(state_robot_or))

        self.observation = np.concatenate((state_robot_pos, state_robot_or,state_object_pos1, state_object_or1,state_object_pos2, state_object_or2,state_object_pos3, state_object_or3),axis=0)
        #print('state',self.observation )

    def get_pose_ob(self,ob_id):
        if (ob_id==1):
            state_object_pos, state_object_or = p.getBasePositionAndOrientation(self.objectid1) #3D pos object;
        elif ob_id==2:
            state_object_pos, state_object_or = p.getBasePositionAndOrientation(self.objectid2) #3D pos object;
        else:
            state_object_pos, state_object_or = p.getBasePositionAndOrientation(self.objectid3) #3D pos object;

        return state_object_pos, state_object_or


    def changeObjPoseCamera(self,position1,orientation1,position2,orientation2,position3,orientation3):
        p.resetBasePositionAndOrientation(self.objectCameraUid1,position1,orientation1)
        p.resetBasePositionAndOrientation(self.objectCameraUid2,position2,orientation2)
        p.resetBasePositionAndOrientation(self.objectCameraUid3,position3,orientation3)
        p.stepSimulation()

    def Removebox(self):
        position2 = [sub_x[2]+2*cell_size,sub_y[0], 0.06]
        orientation2 = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.objectid2,position2,orientation2)

        pos1, or1 = self.get_pose_ob(1)
        pos2, or2 = self.get_pose_ob(2)
        pos3, or3 = self.get_pose_ob(3)
        self.changeObjPoseCamera(pos1+offset,or1,pos2+offset,or2,pos3+offset,or3)
        p.stepSimulation()
        pos_goal = np.array([e_x[1],sub_y[1],0.3],dtype=object)
        self.move(pos_goal,0)


    def check_fingers(self):
        jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque = p.getJointState(self.pandaUid, 9)
        jointPosition2, jointVelocity, jointReactionForces, appliedJointMotorTorque = p.getJointState(self.pandaUid, 10)

        if np.max([jointPosition,jointPosition2]) > 0.01:
            print('Error fingers')
            return 1
        else:
            return 0

    def check_box(self):
        for i in range(1,4):
            pos, orientation = self.get_pose_ob(i)

            if (pos[0]<e_x[0]) or (pos[0]>e_x[-1]) or (pos[1]<e_y[0]) or (pos[1]>e_y[-1]):
                print('Error position - out of region')
                return 1
            else:
                # print('c',np.max([abs(ele) for ele in p.getEulerFromQuaternion(orientation)]))
                if np.max([abs(ele) for ele in p.getEulerFromQuaternion(orientation)])>0.43: #c.a. 25degree
                    print('Error orientation')
                    return 1

        return 0


    def captureImage(self):
        #get the camera image
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=self.w,
                height=self.h,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                lightDiffuseCoeff=100)

        # rgb_array = np.array(rgbImg, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (w,h, 4))
        # rgb_array = np.reshape(rgb_array, (720,960, 4))
        # dim = (256, 256)
        # rgb_array =cv2.resize(rgbImg[:, :, :3], dim, interpolation = cv2.INTER_NEAREST)
        # image = np.array(segImg, dtype=np.uint8)

        im_bgr = cv2.cvtColor(rgbImg, cv2.COLOR_RGBA2BGR)
        im_bgr = cv2.resize(im_bgr, (256,256), interpolation = cv2.INTER_AREA)

        return im_bgr


    def getLabel(self,class_img):
        idx1,idy1 = np.where(class_img==1)

        idx2,idy2 = np.where(class_img==2)

        idx3,idy3 = np.where(class_img==3)

        if (np.absolute(idx1-idx2)==1 and np.absolute(idy1-idy2)==0) or (np.absolute(idx1-idx2)==0 and np.absolute(idy1-idy2)==1) or (np.absolute(idx1-idx3)==1 and np.absolute(idy1-idy3)==0) or (np.absolute(idx1-idx3)==0 and np.absolute(idy1-idy3)==1) or (np.absolute(idx2-idx3)==1 and np.absolute(idy2-idy3)==0) or (np.absolute(idx2-idx3)==0 and np.absolute(idy2-idy3)==1) :
            return 1
        else:
            return 0


    def getClass(self):

        class_img = np.zeros((num1,num2))
        ob_pos , state_object_or = p.getBasePositionAndOrientation(self.objectid1) #3D pos object 1
        # find the class
        idx_x = np.argmin(np.absolute(sub_x - ob_pos[0]))
        idx_y = np.argmin(np.absolute(sub_y - ob_pos[1]))

        class_img[idx_x,idx_y] = 1

        ob_pos , state_object_or = p.getBasePositionAndOrientation(self.objectid2) #3D pos object 2
        # find the class
        idx_x = np.argmin(np.absolute(sub_x - ob_pos[0]))
        idx_y = np.argmin(np.absolute(sub_y - ob_pos[1]))

        class_img[idx_x,idx_y] = 2

        ob_pos , state_object_or = p.getBasePositionAndOrientation(self.objectid3) #3D pos object 3
        # find the class
        idx_x = np.argmin(np.absolute(sub_x - ob_pos[0]))
        idx_y = np.argmin(np.absolute(sub_y - ob_pos[1]))

        class_img[idx_x,idx_y] = 3


        ####### DEBUG: plot the sphere associated to th current class
        # p.resetBasePositionAndOrientation(self.sphereid,[c_x[idx_x],c_y[idx_y],0.2],p.getQuaternionFromEuler([0,0,0]))


        return class_img

    def new_config(self,objectid1Pos,objectid2Pos,objectid3Pos):

        orientation = p.getQuaternionFromEuler([0,0,0*random.uniform(0,0.43/5.)])

        p.resetBasePositionAndOrientation(self.objectid1,objectid1Pos,orientation)
        p.resetBasePositionAndOrientation(self.objectid2,objectid2Pos,orientation)
        p.resetBasePositionAndOrientation(self.objectid3,objectid3Pos,orientation)

        self.changeObjPoseCamera(objectid1Pos+offset,orientation,objectid2Pos+offset,orientation,objectid3Pos+offset,orientation)
        p.stepSimulation()


    def action_fun(self,ob_id,idx2,idy2):

        position = [sub_x[idx2]+random.uniform(-cell_size/8.,cell_size/8.),sub_y[idy2]+random.uniform(-cell_size/8.,cell_size/8.), 0.06]
        orientation = p.getQuaternionFromEuler([0,0,0*random.uniform(0,0.43/5.)])

        if ob_id == 1:
            p.resetBasePositionAndOrientation(self.objectid1,position,orientation)
        elif ob_id == 2:
            p.resetBasePositionAndOrientation(self.objectid2,position,orientation)
        else:
            p.resetBasePositionAndOrientation(self.objectid3,position,orientation)

        pos1, or1 = self.get_pose_ob(1)
        pos2, or2 = self.get_pose_ob(2)
        pos3, or3 = self.get_pose_ob(3)
        self.changeObjPoseCamera(pos1+offset,or1,pos2+offset,or2,pos3+offset,or3)
        p.stepSimulation()





    def noaction_fun(self,class_img):
        idx1,idy1 = np.where(class_img==1)
        pos1 = [sub_x[idx1],sub_y[idy1], 0.06]
        idx2,idy2 = np.where(class_img==2)
        pos2 = [sub_x[idx2],sub_y[idy2], 0.06]
        idx3,idy3 = np.where(class_img==3)
        pos3 = [sub_x[idx3],sub_y[idy3], 0.06]

        self.changeObjPoseCamera(pos1+offset,p.getQuaternionFromEuler([0,0,0]),pos2+offset,p.getQuaternionFromEuler([0,0,0]),pos3+offset,p.getQuaternionFromEuler([0,0,0]))
        p.stepSimulation()


    # def saveImage(self):
    #     #get the camera image
    #     width, height, rgbImg, depthImg, segImg = p.getCameraImage(
    #             width=self.w,
    #             height=self.h,
    #             # width=960,
    #             # height=720,
    #             viewMatrix=view_matrix,
    #             projectionMatrix=proj_matrix,
    #             renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #
    #         # rgb_array = np.array(rgbImg, dtype=np.uint8)
    #         # rgb_array = np.reshape(rgb_array, (w,h, 4))
    #         # rgb_array = np.reshape(rgb_array, (720,960, 4))
    #
    #
    #         # rgb_array = rgb_array[:, :, :3]
    #
    #     image = np.array(segImg, dtype=np.uint8)
    #
    #     hf = h5py.File('camera.h5', 'w')
    #
    #     hf.create_dataset('x_t',data=image)
    #
    #     hf.close()
    #
    #     return image


    def close(self):

        # p.stopStateLogging(self.logId)

        p.disconnect()
