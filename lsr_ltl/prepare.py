import pickle
import random
import numpy as np
import torch





## Create train and test datasets for VAEb
# pkl file name
pkl_filename = "push_v13"
pkl_path = "./datasets/"
pkl_list3=[]

with open(pkl_path + pkl_filename + ".pkl", 'rb') as f:
    pkl_list3 = pickle.load(f)


# suffle the list
random.seed(10)
random.shuffle(pkl_list3)



N_train1 = 0
N_train2 = 3400
N_test1 = 3400
N_test2 = 4000


# print("Train and test datasets sizes: ", N_train2-N_train1,N_test2-N_test1)

pkl_train3 = pkl_list3[N_train1:N_train2]
pkl_test3 = pkl_list3[N_test1:N_test2]


pkl_filename = "push_v12"
pkl_path = "./datasets/"
pkl_list2=[]

with open(pkl_path + pkl_filename + ".pkl", 'rb') as f:
    pkl_list2 = pickle.load(f)

with open(pkl_path + "push_v1" + ".pkl", 'wb') as f:
    pickle.dump(pkl_list2+pkl_list3, f)

# suffle the list
random.seed(10)
random.shuffle(pkl_list2)



N_train1 = 0
N_train2 = 1200
N_test1 = 1200
N_test2 = 1500



pkl_train2 = pkl_list2[N_train1:N_train2]
pkl_test2 = pkl_list2[N_test1:N_test2]


pkl_train = pkl_train2 + pkl_train3
pkl_test = pkl_test2 + pkl_test3

pkl_list = None


# Write test pkl file
test_data = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                            torch.tensor(t[1]/255).float().permute(2, 0, 1),
                            torch.tensor(t[2]).float()),
                pkl_test))

with open(pkl_path + 'test_' + "push_v1" + ".pkl", 'wb') as f:
    pickle.dump(test_data, f)



print("test data saved")

test_data = None





# Write train pkl file
train_data = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                            torch.tensor(t[1]/255).float().permute(2, 0, 1),
                            torch.tensor(t[2]).float()),
                pkl_train))




with open(pkl_path + 'train_' + "push_v1" + ".pkl", 'wb') as f:
    pickle.dump(train_data, f)



print("train data saved")



print("Train and test datasets sizes: ", len(pkl_train),len(pkl_test))

train_data = None
