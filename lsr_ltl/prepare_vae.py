import pickle
import random
import numpy as np
import torch





## Create train and test datasets for VAE
# pkl file name
pkl_filename = "push_v1_vae"
pkl_path = "./datasets/"
pkl_list=[]

with open(pkl_path + pkl_filename + ".pkl", 'rb') as f:
    pkl_list = pickle.load(f)


# suffle the list
random.seed(10)
random.shuffle(pkl_list)


length = len(pkl_list)
N_train1 = 0
N_train2 = 5500
N_test1 = N_train2
N_test2 = 7000


print("Train and test datasets sizes: ", N_train2-N_train1,N_test2-N_test1)

pkl_train = pkl_list[N_train1:N_train2]
pkl_test = pkl_list[N_test1:N_test2]




pkl_list = None


# for i,elem in enumerate(pkl_test):
#     print(elem[2])

# Write test pkl file
test_data = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                            torch.tensor(t[1]/255).float().permute(2, 0, 1),
                            torch.tensor(t[2]).float()),
                pkl_test))

with open(pkl_path + 'test_' + pkl_filename + ".pkl", 'wb') as f:
    pickle.dump(test_data, f)



print("test data saved")

test_data = None





# Write train pkl file
train_data = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                            torch.tensor(t[1]/255).float().permute(2, 0, 1),
                            torch.tensor(t[2]).float()),
                pkl_train))




with open(pkl_path + 'train_' + pkl_filename + ".pkl", 'wb') as f:
    pickle.dump(train_data, f)


print("train data saved")


train_data = None
