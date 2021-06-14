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


if ltl == "adjacent":
    state = './state_dict_adj.pt'
    num_elem = 4000
    hidden_dim = 12
    n_layers = 4
    batch_size = 200


    X_pickle = pickle.load( open( "./datasets_lstm/X_a_all.pkl", "rb" ) )
    Y_pickle = pickle.load( open( "./datasets_lstm/Y_a_all.pkl", "rb" ) )





XY = list(zip(X_pickle, Y_pickle))

random.Random(10).shuffle(XY)

X_pickle, Y_pickle = zip(*XY)


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
#
# exit(0)


X_test = np.array(X_pickle)
Y_test = np.array(Y_pickle)

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



X_test,Y_test, lengths_test = sort_and_pad(X_test,Y_test)







import torch
from torch.utils.data import TensorDataset, DataLoader

# print(X_train[0])
# exit(0)

test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test), torch.from_numpy(lengths_test))



test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)




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

class LTLNet(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=.6):
        super(LTLNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True) # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, lens):
        batch_size = x.size(0)
        # x = x.long()

        # print(hidden[0].size())
        # input("Press")

        # print(hidden[0].type())

        # print(x[0][0][0])
        # print(x[1][0][0])


        # # Packs a Tensor containing padded sequences of variable length. If batch_first is True, B x T x * input is expected.
        # x = pack_padded_sequence(x, lens, batch_first = True, enforce_sorted = True) # unpad

        # print(x[0][0][0])
        # print(x[0][1][0])
        # exit(0)

        lstm_out, hidden = self.lstm(x, hidden)

        # # Pads a packed batch of variable length sequences.
        # # It is an inverse operation to pack_padded_sequence().
        # lstm_out, lens2 = pad_packed_sequence(lstm_out, batch_first=True) # pad the sequence to the max length in the batch

        # print(lstm_out)
        # exit(0)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden



output_size = 1
embedding_dim = 64
# hidden_dim = 12
# n_layers = 2

model = LTLNet(output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)
print(model)



lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# epochs = 500
counter = 0
print_every = 1
clip = 5
valid_loss_min = np.Inf




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
    h = model.init_hidden(len(inputs))
    h = tuple([each.data for each in h])
    inputs, labels, lens = inputs.to(device), labels.to(device), lens.to(device)
    output, h = model(inputs, h, lens)
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
print("Test accuracy: {:.3f}".format(test_acc))


# test_acc_neg = num_correct_neg/num_neg
# test_acc_pos = num_correct_pos/num_pos
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
