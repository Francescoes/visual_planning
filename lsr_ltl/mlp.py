import bz2
from collections import Counter
import re
import nltk
import numpy as np
nltk.download('punkt')


import pickle
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random



ltl = "adjacent"
# ltl = "row"

if ltl == "row":
    state = './mlp_state_dict_row.pt'
    num_elem = 2000

    X_pickle = pickle.load( open( "./datasets_lstm/X.pkl", "rb" ) )
    Y_pickle = pickle.load( open( "./datasets_lstm/Y.pkl", "rb" ) )

if ltl == "adjacent":
    state = './mlp_state_dict_adj.pt'
    num_elem = 4000

    X_pickle = pickle.load( open( "./datasets_lstm/X_a.pkl", "rb" ) )
    Y_pickle = pickle.load( open( "./datasets_lstm/Y_a.pkl", "rb" ) )


epochs = 100
batch_size = 200


XY = list(zip(X_pickle, Y_pickle))

random.Random(10).shuffle(XY)

X_pickle, Y_pickle = zip(*XY)


# num_elem = len(Y_pickle)
print("Dataset size: ",num_elem)
# exit(0)
X_pickle = X_pickle[:num_elem]
Y_pickle = Y_pickle[:num_elem]

#
# print((X_pickle[0][1]))
# print(Y_pickle)
# exit(0)

print("Percentage of valid/not valid paths: ",Y_pickle.count(1)/len(Y_pickle)*100,"%", Y_pickle.count(0)/len(Y_pickle)*100,"%")


# import seaborn as sns
#
#
# fig = plt.figure(figsize=(8,5))
# ax = sns.barplot(x=list(set(Y_pickle)),y=[Y_pickle.count(0),Y_pickle.count(1)]);
# ax.set(xlabel='Valid')
# ax.set(ylabel='Number of paths that satisfy/not satisfy LTLs')
#
# plt.show()
#
#
#
# fig = plt.figure(figsize=(8,4))
# ax = sns.distplot([len(x) for x in X_pickle],kde=False);
# ax.set(xlabel='Path Length', ylabel='Frequency')
#
# plt.show()

# exit(0)


# 70% training
# 20% testing
# 10% validation

split_frac1 = 0.7
split_id1 = int(split_frac1 * len(X_pickle))
split_frac2 = 0.8
split_id2 = int(split_frac2 * len(X_pickle))

X_train, X_test, X_val = np.array(X_pickle[:split_id1]), np.array(X_pickle[split_id1:split_id2]), np.array(X_pickle[split_id2:])
Y_train, Y_test, Y_val = np.array(Y_pickle[:split_id1]), np.array(Y_pickle[split_id1:split_id2]), np.array(Y_pickle[split_id2:])

# lengths_train, lengths_test, lengths_val = np.array(lengths[:split_id1]), np.array(lengths[split_id1:split_id2]), np.array(lengths[split_id2:])

# print(len(X_train[0]))
# exit(0)





def pad_seq(s,maxlen,emb_dim):
    padded = np.zeros((maxlen,emb_dim),dtype=np.float32)
    padded[:len(s)] = s
    return list(padded)


def pad_dataset(X):

    maxlen = max([len(x) for x in X])
    emb_dim = len(X[0][0])
    for i,s in enumerate(X):
        X[i] = pad_seq(s,maxlen,emb_dim)


    return X

def sort_and_pad(X_pickle,Y_pickle):

    X_sorted, Y_sorted = zip(*sorted(zip(X_pickle, Y_pickle),key=lambda x: len(x[0]), reverse = True))

    X_sorted = list(X_sorted)
    Y_sorted = list(Y_sorted)


    lengths = [len(x) for x in X_sorted] # lengths after sorting

    X = pad_dataset(X_sorted)

    return np.array(X_sorted),np.array(Y_sorted),np.array(lengths)



X_train,Y_train, lengths_train = sort_and_pad(X_train,Y_train)
X_test,Y_test, lengths_test = sort_and_pad(X_test,Y_test)
X_val,Y_val, lengths_val = sort_and_pad(X_val,Y_val)




# X = [(X_pad[i],lengths[i]) for i in range(len(X_pad))]


# print(X[0])
# exit(0)
# print(X)
# exit(0)

# print(X[4][0][0].type)




import torch
from torch.utils.data import TensorDataset, DataLoader

# print(X_train[0])
# exit(0)

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(lengths_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test), torch.from_numpy(lengths_test))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val), torch.from_numpy(lengths_val))





# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")





import torch.nn as nn

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(8 * 64, 4*64),
          nn.ReLU(),
          nn.Linear(4 * 64, 2*64),
          nn.ReLU(),
          nn.Linear(2 * 64, 32),
          nn.ReLU(),
          nn.Linear(32, 4),
          nn.ReLU(),
          nn.Linear(4, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)









# epochs = 500
counter = 0
print_every = 1
clip = 5




accuracy_list = []
precision_list = []
recall_list = []
specificity_list = []
F_list = []

K = 10
for k in range(K):

    model = mlp()
    model.to(device)
    # print(model)

    lr=0.005
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    valid_loss_min = np.Inf

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)



    train_loss = [] # train loss
    validation_loss = [] # validation loss


    train_loss_epochs = [] # train loss every epoch
    validation_loss_epochs = [] # validation loss every epoc

    val_losses = []
    model.eval()
    counter=0

    for inp, lab, le in val_loader:
        counter += 1
        inp, lab = inp.to(device), lab.to(device)
        out = model(inp)
        val_loss = criterion(out.squeeze(), lab.float())
        val_losses.append(val_loss.item())

        validation_loss.append(val_loss)


    model.train()
    for i in range(epochs):
        model.train()

        for inputs, labels, lens in train_loader:
            counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()

            output = model(inputs)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss.append(loss)

        counter = 0
        # val_losses = []
        model.eval()
        for inp, lab, le in val_loader:
            counter += 1
            inp, lab = inp.to(device), lab.to(device)
            out = model(inp)
            val_loss = criterion(out.squeeze(), lab.float())
            val_losses.append(val_loss.item())

            validation_loss.append(val_loss)




        if i%print_every==0:
            print("\nEpoch: {}/{}...".format(i+1, epochs),"Loss: {:.6f}...".format(loss.item()),"Val Loss: {:.6f}".format(np.mean(val_losses)))

            validation_loss_epochs.append(np.mean(val_losses))
            train_loss_epochs.append(loss)

            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), state)
                print('\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)



    val_losses = []
    model.eval()
    for inp, lab, le in val_loader:
        inp, lab = inp.to(device), lab.to(device)
        out = model(inp)
        val_loss = criterion(out.squeeze(), lab.float())
        val_losses.append(val_loss.item())

        validation_loss.append(val_loss)

        print("\nVal Loss: {:.6f}".format(np.mean(val_losses)))
        if np.mean(val_losses) <= valid_loss_min:
            torch.save(model.state_dict(), state)
            print('\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
            valid_loss_min = np.mean(val_losses)




    # #Loading the best model
    model.load_state_dict(torch.load(state))








    test_losses = []
    num_correct = 0



    num_correct_neg = 0
    num_correct_pos = 0
    num_not_correct_pos = 0
    num_not_correct_neg = 0
    num_neg = 0
    num_pos = 0


    model.eval()
    for inputs, labels, lens in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze()) #rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


        for k in range(len(pred)):
            if labels[k]==0:
                num_neg+=1
            else:
                num_pos+=1

            if pred[k]==labels[k]:
                if pred[k]==0:
                    num_correct_neg+=1
                else:
                    num_correct_pos+=1
            else:
                if pred[k]==0:
                    num_not_correct_neg+=1
                else:
                    num_not_correct_pos+=1



    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(test_loader.dataset)
    # print("Test accuracy: {:.3f}%".format(test_acc*100))
    print("Test accuracy: {:.3f}".format(test_acc))


    test_acc_neg = num_correct_neg/num_neg
    test_acc_pos = num_correct_pos/num_pos
    # print("Test accuracy negatives: {:.3f}%".format(test_acc_neg*100))
    # print("Test accuracy positives: {:.3f}%".format(test_acc_pos*100))


    # tp = num_correct_pos
    # tn = num_correct_neg

    # fp = num_not_correct_pos
    # fn = num_not_correct_neg

    # Precision ( positive predictive value (PPV)) tp/(tp+fp)
    precision = num_correct_pos/(num_correct_pos + num_not_correct_pos)

    # Recall (true positive rate or sensitivity)  tp/(tp+fn)
    recall = num_correct_pos/(num_correct_pos + num_not_correct_neg)

    # Specificity (True negative rate) tn/(tn+fp)
    specificity = num_correct_neg/(num_correct_neg + num_not_correct_pos)

    # A measure that combines precision and recall is the harmonic mean of precision and recall, the traditional F-measure or balanced F-score:
    # F = 2 * (precision*recall)/(precision+recall)
    F = 2 * (precision*recall)/(precision+recall)


    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("Specificity: {:.3f}".format(specificity))
    print("F: {:.3f}".format(F))



    accuracy_list.append(test_acc)
    precision_list.append(precision)
    recall_list.append(recall)
    specificity_list.append(specificity)
    F_list.append(F)


    # e = list(range(len(train_loss_epochs)))
    #
    # plt.figure()
    # plt.plot(e, np.array(train_loss_epochs), 'bo--')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.title('Training loss')
    #
    # plt.grid(True)
    # plt.show()
    #
    #
    #
    # plt.plot(e, np.array(validation_loss_epochs), 'go-')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.title('Validation loss')
    #
    # plt.grid(True)
    # plt.show()


from statistics import mean

print("Mean Test Accuracy: {:.3f}".format(mean(accuracy_list)))
print("Mean Precision: {:.3f}".format(mean(precision_list)))
print("Mean Recall: {:.3f}".format(mean(recall_list)))
print("Mean Specificity: {:.3f}".format(mean(specificity_list)))
print("Mean F: {:.3f}".format(mean(F_list)))
