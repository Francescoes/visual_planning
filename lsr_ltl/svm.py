import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import random



ltl = "adjacent"
# ltl = "row"

if ltl == "row":
    state = './state_dict_row.pt'
    num_elem = 2000


    X_pickle = pickle.load( open( "./datasets_lstm/X.pkl", "rb" ) )
    Y_pickle = pickle.load( open( "./datasets_lstm/Y.pkl", "rb" ) )

if ltl == "adjacent":
    state = './state_dict_adj.pt'
    num_elem = 4000


    X_pickle = pickle.load( open( "./datasets_lstm/X_a.pkl", "rb" ) )
    Y_pickle = pickle.load( open( "./datasets_lstm/Y_a.pkl", "rb" ) )




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


X_train = np.reshape(X_train, (-1, 8*64))
X_test = np.reshape(X_test, (-1, 8*64))
X_val = np.reshape(X_val, (-1, 8*64))







model_linear = SVC(kernel='linear')
model_linear.fit(X_train, Y_train)

# predict
Y_pred = model_linear.predict(X_test)



# confusion matrix and accuracy

from sklearn import metrics
from sklearn.metrics import confusion_matrix
# accuracy
print("Test Accuracy:", metrics.accuracy_score(y_true=Y_test, y_pred=Y_pred))

# cm
cm = metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred)
tn, fp, fn, tp = cm.ravel()
# print(cm)




# Precision ( positive predictive value (PPV)) tp/(tp+fp)
precision = tp/(tp + fp)

# Recall (true positive rate or sensitivity)  tp/(tp+fn)
recall = tp/(tp + fn)

# Specificity (True negative rate) tn/(tn+fp)
specificity = tn/(tn + fp)

# A measure that combines precision and recall is the harmonic mean of precision and recall, the traditional F-measure or balanced F-score:
# F = 2 * (precision*recall)/(precision+recall)
F = 2 * (precision*recall)/(precision+recall)


print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("Specificity: {:.3f}".format(specificity))
print("F: {:.3f}".format(F))



# non-linear model
# using rbf kernel, C=1, default value of gamma

# model
non_linear_model = SVC(kernel='rbf')

# fit
non_linear_model.fit(X_train, Y_train)

# predict
Y_pred = non_linear_model.predict(X_test)




# confusion matrix and accuracy

# accuracy
print("Test Accuracy:", metrics.accuracy_score(y_true=Y_test, y_pred=Y_pred))

# cm
cm = metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred)
tn, fp, fn, tp = cm.ravel()
# print(cm)




# Precision ( positive predictive value (PPV)) tp/(tp+fp)
precision = tp/(tp + fp)

# Recall (true positive rate or sensitivity)  tp/(tp+fn)
recall = tp/(tp + fn)

# Specificity (True negative rate) tn/(tn+fp)
specificity = tn/(tn + fp)

# A measure that combines precision and recall is the harmonic mean of precision and recall, the traditional F-measure or balanced F-score:
# F = 2 * (precision*recall)/(precision+recall)
F = 2 * (precision*recall)/(precision+recall)


print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("Specificity: {:.3f}".format(specificity))
print("F: {:.3f}".format(F))
