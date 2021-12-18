#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 22:49:09 2021

@author: pnaddaf
"""

import mydatasets
import mymodels
import utils
import numpy as np
import torch
import sys
import os
import json 
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics._ranking import _binary_clf_curve, precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, plot_roc_curve
from matplotlib import pyplot as plt


# X = pd.read_csv('../data/gene_data.csv', header=None, index_col=0)

# Y = pd.read_csv('../data/AllLabels.csv', index_col='id')
# Y = Y[Y.index.isin(X.index)]

# NoYs = X[np.logical_not(X.index.isin(Y.index))]
# print('Xs with no Y found:', NoYs.shape[0])
# X = X.drop(NoYs.index)

# X = X.sort_index().reset_index().rename(columns={0: 'id'})
# Y = Y.sort_index().reset_index()

# X_mat = X.iloc[:, 1:].to_numpy()
# Y_mat = Y.iloc[:, 1:].to_numpy()

# val_test_count = int(X_mat.shape[0] * 0.1)
# val_test_indcs = (np.random.permutation(X_mat.shape[0])[:2*val_test_count]).reshape((2, -1))

# X_val = X_mat[val_test_indcs[0], :]
# Y_val = Y_mat[val_test_indcs[0], :]

# X_test = X_mat[val_test_indcs[1], :]
# Y_test = Y_mat[val_test_indcs[1], :]

# X_train = np.delete(X_mat, val_test_indcs.flatten(), axis=0)
# Y_train = np.delete(Y_mat, val_test_indcs.flatten(), axis=0)
# print(f'Train Count: {X_train.shape[0]}')
# print(f'Validation Count: {X_val.shape[0]}')
# print(f'Test Count: {X_test.shape[0]}')

# shuffle_indcs = np.random.permutation(X_train.shape[0])
# X_train = X_train[shuffle_indcs, :]
# Y_train = Y_train[shuffle_indcs, :]


def get_roc(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, model, n_neighbors=5, model_name="FC"):
    if model_name == "LRCN":

        xx_train, yy_train, fs = prepare_data(x_train, y_train)
        xx_test, yy_test, fs = prepare_data(x_eval, y_eval)
        emb_train = model(torch.from_numpy(xx_train.astype('float32'))).detach().numpy()
        emb_test = model(torch.from_numpy(xx_test.astype('float32'))).detach().numpy()
    else:
        emb_train = model(torch.from_numpy(x_train.astype('float32'))).detach().numpy()
        emb_val = model(torch.from_numpy(x_eval.astype('float32'))).detach().numpy()   

    KNN = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(emb_train)

    nbr_dists, nbr_indcs = KNN.kneighbors(emb_val)
    y_k_neghbors = y_train[nbr_indcs, :]
    y_pred = np.nanmean(y_k_neghbors, axis=1)
    y_pred[np.where(np.isnan(y_pred))] = 0.5
    #y_pred = np.round(y_pred)

    condition = np.where(np.logical_and(np.logical_not(np.isnan(y_eval)), np.logical_not(np.isnan(y_pred))))

    #### CALCULATE ROC

    #return np.sum(y_pred[condition] == y_eval[condition]) / y_eval[condition].shape[0]
    
    

def get_acc(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, model, n_neighbors=5, model_name="FC"):
    if model_name == "LRCN":

        xx_train, yy_train, fs = prepare_data(x_train, y_train)
        xx_test, yy_test, fs = prepare_data(x_eval, y_eval)
        emb_train = model(torch.from_numpy(xx_train.astype('float32'))).detach().numpy()
        emb_val = model(torch.from_numpy(xx_test.astype('float32'))).detach().numpy()
        print(xx_train.shape)
        print(emb_train.shape)
    else:
        emb_train = model(torch.from_numpy(x_train.astype('float32'))).detach().numpy()
        emb_val = model(torch.from_numpy(x_eval.astype('float32'))).detach().numpy() 
  

    y_pred = emb_val
        
    KNN = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(emb_train)

    nbr_dists, nbr_indcs = KNN.kneighbors(emb_val)
    y_k_neghbors = y_train[nbr_indcs, :] #y_eval x 5 x 12
    y_pred = np.nanmean(y_k_neghbors, axis=1) #y_eval x 12
    y_pred = np.round(y_pred) #0.5 > --> 1

    condition = np.where(np.logical_and(np.logical_not(np.isnan(y_eval)), np.logical_not(np.isnan(y_pred))))

    return np.sum(y_pred[condition] == y_eval[condition]) / y_eval[condition].shape[0]



def ROC_Score(model, X_test, Y_test,X_train, Y_train, limited=False, model_name = "FC"):
    num_of_drugs = 12
    
    if model_name == "LRCN":

        xx_train, yy_train, fs = prepare_data(X_train, Y_train)
        xx_test, yy_test, fs = prepare_data(X_test, Y_test)
        emb_train = model(torch.from_numpy(xx_train.astype('float32'))).detach().numpy()
        emb_test = model(torch.from_numpy(xx_test.astype('float32'))).detach().numpy()
    else:
        emb_train = model(torch.from_numpy(X_train.astype('float32'))).detach().numpy()
        emb_test = model(torch.from_numpy(X_test.astype('float32'))).detach().numpy() 
        
    KNN = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(emb_train)


    nbr_dists, nbr_indcs = KNN.kneighbors(emb_test)
    y_k_neghbors = Y_train[nbr_indcs, :] #y_eval x 5 x 12
    y_pred_keras_tmp = np.nanmean(y_k_neghbors, axis=1) #y_eval x 1
    
    condition_nan =  np.where(np.isnan(y_pred_keras_tmp))
    y_pred_keras_tmp[condition_nan] = 0.5
    
    condition = np.where(np.isnan(Y_test))
    Y_test[condition] = -1
    print(Y_test)

    
    for i in range(0, num_of_drugs):
        y_test_tmp = Y_test[:, i]
        y_pred_keras = y_pred_keras_tmp[:, i]
        i2 = 0
        while i2 < len(y_test_tmp):
            if y_test_tmp[i2] != 0 and y_test_tmp[i2] != 1:
                y_test_tmp = np.delete(y_test_tmp, i2)
                y_pred_keras = np.delete(y_pred_keras, i2)
            else:
                i2 = i2 + 1
    y_test_tmp = []
    y_pred_keras = []
    for i in range(0, num_of_drugs):
        y_test_tmp.extend(Y_test[:, i])
        y_pred_keras.extend(y_pred_keras_tmp[:, i])
    i = 0
    while i < len(y_test_tmp):
        if y_test_tmp[i] != 0 and y_test_tmp[i] != 1:
            y_test_tmp = np.delete(y_test_tmp, i)
            y_pred_keras = np.delete(y_pred_keras, i)
        else:
            i = i + 1
    fpr_keras, tpr_keras, _ = roc_curve(y_test_tmp, y_pred_keras)
    # print("___")
    # print(fpr_keras)
    # print("___")
    # print(tpr_keras)
    # print("___")
    auc_keras = auc(fpr_keras, tpr_keras)
    # print(auc_keras)
    return auc_keras




def ROC(model, X_test, y_test, X_train, Y_train, name, multi=False, limited=False, bccdc=False, model_name = "FC"):
    
    num_of_drugs = 12
    if model_name == "LRCN":

        xx_train, yy_train, fs = prepare_data(X_train, Y_train)
        xx_test, yy_test, fs = prepare_data(X_test, Y_test)
        emb_train = model(torch.from_numpy(xx_train.astype('float32'))).detach().numpy()
        emb_test = model(torch.from_numpy(xx_test.astype('float32'))).detach().numpy()
    else:
        emb_train = model(torch.from_numpy(X_train.astype('float32'))).detach().numpy()
        emb_test = model(torch.from_numpy(X_test.astype('float32'))).detach().numpy() 
    

    KNN = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(emb_train)


    nbr_dists, nbr_indcs = KNN.kneighbors(emb_test)
    y_k_neghbors = Y_train[nbr_indcs, :] #y_eval x 5 x 12
    y_pred_keras_tmp = np.nanmean(y_k_neghbors, axis=1) #y_eval x 1
    
    condition_nan =  np.where(np.isnan(y_pred_keras_tmp))
    y_pred_keras_tmp[condition_nan] = 0.5
    
    condition = np.where(np.isnan(y_test))
    y_test[condition] = -1
    print(y_test)    
    
    y_pred_keras = []
    y_test_tmp = []
    scores = []
    if limited:
        num_of_drugs = 7


    if bccdc:
        num_of_drugs = 5

        tmp_b = []
        for i in range(0, len(y_pred_keras_tmp)):
            tmp_b_b = [y_pred_keras_tmp[i][0], y_pred_keras_tmp[i][1], y_pred_keras_tmp[i][2], y_pred_keras_tmp[i][6],
                       y_pred_keras_tmp[i][8]]
            tmp_b.append(tmp_b_b)

        y_pred_keras_tmp = np.array(tmp_b)

    if multi == False:
        for i in range(0, len(y_pred_keras_tmp)):
            y_pred_keras.append(y_pred_keras_tmp[i][1])
            y_test_tmp.append(y_test[i][1])
        ROC_maker(y_test_tmp, y_pred_keras, name)
    else:
        for i in range(0, num_of_drugs):  # len(y_test[0])):
            y_test_tmp = y_test[:, i]
            y_pred_keras = y_pred_keras_tmp[:, i]
            # bug? cahnge i2 to i
            i2 = 0
            while i2 < len(y_test_tmp):
                if y_test_tmp[i2] != 0 and y_test_tmp[i2] != 1:
                    y_test_tmp = np.delete(y_test_tmp, i2)
                    y_pred_keras = np.delete(y_pred_keras, i2)
                else:
                    i2 = i2 + 1
            try:
                # print(len(y_test_tmp))
                # print(len(y_test_tmp[0]))
                # print(len(y_pred_keras))
                # print(len(y_pred_keras[0]))
                if i != 0:
                    if i < num_of_drugs - 1:
                        scores.append(ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), False, False))
                    else:
                        scores.append(ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), False, True))
                else:
                    scores.append(ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), True, False))

            except():
                print("error on " + i + " " + y_test_tmp)
        y_test_tmp = []
        y_pred_keras = []
        for i in range(0, num_of_drugs):  # len(y_test[0])):
            y_test_tmp.extend(y_test[:, i])
            # print(y_test_tmp)
            y_pred_keras.extend(y_pred_keras_tmp[:, i])
        i = 0
        while i < len(y_test_tmp):
            if y_test_tmp[i] != 0 and y_test_tmp[i] != 1:
                y_test_tmp = np.delete(y_test_tmp, i)
                y_pred_keras = np.delete(y_pred_keras, i)
            else:
                i = i + 1
        ROC_maker(y_test_tmp, y_pred_keras, name + " _ All", True)
        # fpr_keras, tpr_keras, _ = roc_curve(y_test_tmp, y_pred_keras)
        # auc_keras = auc(fpr_keras, tpr_keras)
        # print(auc_keras)
        return scores
    

def specificity_recall_calculator(y_true, probas_pred, pos_label=None,
                                  sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)

    specificity = (fps[-1] - fps) / fps[-1]
    specificity[np.isnan(specificity)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[specificity[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def SR_maker(y_test_tmp, y_pred_keras):
    specificity, recall, th = specificity_recall_calculator(y_test_tmp, y_pred_keras)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test_tmp, y_pred_keras)
    score = 0
    count = 0
    for i in range(0 ,len(recall)):
        if specificity[i] == 0.95:
            score += recall[i]
            count += count + 1

    if score != 0:
        return (score/count), auc(lr_recall, lr_precision)

    for i in range(0 ,len(recall)):
        if specificity[i] <= 0.952 and specificity[i] >= 0.945:
            score += recall[i]
            count += 1

    if score != 0:
        return (score/count), auc(lr_recall, lr_precision)

    for i in range(0, len(recall)):
        if specificity[i] <= 0.955 and specificity[i] >= 0.940:
            score += recall[i]
            count += 1
    if score != 0:
        return (score / count), auc(lr_recall, lr_precision)
    else:
        return 0, auc(lr_recall, lr_precision)

def PR(model, X_test, y_test, X_train, Y_train, bccdc=False, model_name="FC"):
    num_of_drugs = 12
    if model_name == "LRCN":

        xx_train, yy_train, fs = prepare_data(X_train, Y_train)
        xx_test, yy_test, fs = prepare_data(X_test, Y_test)
        emb_train = model(torch.from_numpy(xx_train.astype('float32'))).detach().numpy()
        emb_test = model(torch.from_numpy(xx_test.astype('float32'))).detach().numpy()
    else:
        emb_train = model(torch.from_numpy(X_train.astype('float32'))).detach().numpy()
        emb_test = model(torch.from_numpy(X_test.astype('float32'))).detach().numpy()        
        
    KNN = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(emb_train)


    nbr_dists, nbr_indcs = KNN.kneighbors(emb_test)
    y_k_neghbors = Y_train[nbr_indcs, :] #y_eval x 5 x 12
    y_pred_keras_tmp = np.nanmean(y_k_neghbors, axis=1) #y_eval x 1
    
    condition_nan =  np.where(np.isnan(y_pred_keras_tmp))
    y_pred_keras_tmp[condition_nan] = 0.5
    
    condition = np.where(np.isnan(y_test))
    y_test[condition] = -1

    y_pred_keras = []
    y_test_tmp = []
    scores_sr = []
    scores_pr = []

    if bccdc:
        num_of_drugs = 5

        tmp_b = []
        for i in range(0, len(y_pred_keras_tmp)):
            tmp_b_b = [y_pred_keras_tmp[i][0], y_pred_keras_tmp[i][1], y_pred_keras_tmp[i][2], y_pred_keras_tmp[i][6],
                       y_pred_keras_tmp[i][8]]
            tmp_b.append(tmp_b_b)

        y_pred_keras_tmp = np.array(tmp_b)



    for i in range(0, num_of_drugs):  # len(y_test[0])):
        y_test_tmp = y_test[:, i]
        y_pred_keras = y_pred_keras_tmp[:, i]
        i2 = 0
        while i2 < len(y_test_tmp):
            if y_test_tmp[i2] != 0 and y_test_tmp[i2] != 1:
                y_test_tmp = np.delete(y_test_tmp, i2)
                y_pred_keras = np.delete(y_pred_keras, i2)
            else:
                i2 = i2 + 1
        try:
            if i != 0:
                if i < num_of_drugs - 1:
                    sr, pr = SR_maker(y_test_tmp, y_pred_keras)
                    scores_sr.append(sr)
                    scores_pr.append(pr)
                else:
                    sr, pr = SR_maker(y_test_tmp, y_pred_keras)
                    scores_sr.append(sr)
                    scores_pr.append(pr)
            else:
                sr, pr = SR_maker(y_test_tmp, y_pred_keras)
                scores_sr.append(sr)
                scores_pr.append(pr)

        except():
            print("error on " + i + " " + y_test_tmp)
    return scores_sr, scores_pr




def ROC_maker(y_test_tmp, y_pred_keras, name, clear=True, save=True):
    # print(y_test_tmp)
    fpr_keras, tpr_keras, _ = roc_curve(y_test_tmp, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    if clear:
        plt.clf()
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve _ ' + name)
    plt.legend(loc='best')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    # if save:
    #     fig1.savefig('result/ROC_' + name + '.png', dpi=100)
    return auc_keras



import torch.nn as nn
import torch.nn.functional as F
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=8, kernel_size=3, padding = 'same')
        self.act = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv1d(8, 4, kernel_size=6, padding = 'same')
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=1)
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=518, batch_first=True, num_layers=3, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=518, hidden_size=64, batch_first=True, num_layers=3, dropout=0.1)
        self.dense1 = nn.Linear(64, 64)
        self.dense2= nn.Linear(64, 518)
        self.dense3 = nn.Linear(518, 12)
    

    def forward(self, x):
        x = self.drop1(x)
        x = self.conv1(x.permute(0,2,1))
        x = self.act(x)  
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)
        x , h  = self.lstm1(x.permute(0,2,1))
        # x = self.drop1(x)
        x , h = self.lstm2(x)
        x = self.drop2(x[:,-1,:])
        # x = x.reshape(x.size(0), -1)
        x = self.act(self.dense1(x))
        x = self.drop3(x)
        x = self.act(self.dense2(x))
        # x = torch.sigmoid(self.dense3(x))
        

        return x
# class NeuralNet(nn.Module):
#     def __init__(self):
#         super(NeuralNet, self).__init__()
#         self.drop1 = nn.Dropout(0.1)
#         self.drop2 = nn.Dropout(0.1)
#         self.drop3 = nn.Dropout(0.1)
#         self.conv1 = nn.Conv1d(in_channels=20, out_channels=8, kernel_size=3, padding = 'same')
#         self.act = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
#         self.conv2 = nn.Conv1d(8, 4, kernel_size=6, padding = 'same')
#         self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=1)
#         # self.lstm1 = nn.LSTM(input_size=4, hidden_size=518, batch_first=True, num_layers=3, dropout=0.1)
#         # self.lstm2 = nn.LSTM(input_size=518, hidden_size=64, batch_first=True, num_layers=3, dropout=0.1)
#         self.dense1 = nn.Linear(200*20, 1500)
#         self.dense2= nn.Linear(1500, 518)
#         self.dense3 = nn.Linear(518, 12)
#     def forward(self, x):
#         # x = self.drop1(x)
#         # x = self.conv1(x.permute(0,2,1))
#         # x = self.act(x)  
#         # x = self.pool1(x)
#         # x = self.conv2(x)
#         # x = self.act(x)
#         # x = self.pool2(x)
#         # x , h  = self.lstm1(x.permute(0,2,1))
#         # # x = self.drop1(x)
#         # x , h = self.lstm2(x)
#         x = self.drop2(x.reshape(-1,200*20))
#         # x = x.reshape(x.size(0), -1)
#         x = self.act(self.dense1(x))
#         x = self.drop3(x)
#         x = self.act(self.dense2(x))
#         x = self.dense3(x)
#         # x = torch.sigmoid(self.dense3(x))

#         return x
    
def prepare_data(features, label):
    # TODO
    FrameSize = 200
    X = features.tolist()
    y = label.tolist()

    for i in range(0, len(X)):
        if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
            for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
                X[i].append(0)
        X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))

    X = np.array(X)
    y = np.array(y)
    return X, y, FrameSize



def masked_loss_function(y_true, y_pred):   
    condition = np.where(np.logical_not(np.isnan(y_true)))

    loss = nn.BCELoss()

    y_true = torch.tensor(y_true[condition]).float()
    return loss( y_pred[condition] ,y_true)

# factors=np.zeros(Y_train.shape[1])+1.0
# log_every=1
# model_name = "LRCN"
# if model_name == "LRCN":
#     model = NeuralNet()
# else:
#     model=mymodels.SimpleNet(X_train.shape[1], 30, [X_train.shape[1], 1500, 30])
    
# for epoch in range(100):
#     # get scheduled values of hyper params
#     tmargin=0
#     batch_size=500
#     lrate=0.001
#     max_trips=100
#     max_neg=3
#     print("Epoch ",epoch,(tmargin,batch_size,lrate,max_trips,max_neg))
#     # define loss and create optimizer
#     triplet_loss = torch.nn.TripletMarginLoss(margin=tmargin, p=2)
#     optimizer = torch.optim.Adam(model.parameters(),lr=lrate)
#     # get batches
#     mini_batches=utils.make_batches(X_train, Y_train, batch_size)
#     loss_values=[]
#     for batch_num,batch in enumerate(tqdm(mini_batches, leave=False)):
#         x_batch,y_batch=batch
#         # x_batch = X_train 
#         # y_batch = Y_train
#         if model_name == "LRCN":
#         # generate embeddings
#             xx, yy, fs = prepare_data(x_batch, y_batch)
#             embeddings = model(torch.tensor(xx.astype('float32')))
#         else:
#             embeddings =model(torch.from_numpy(x_batch.astype('float32')))
#         # generate triplets (online)
#         trips=utils.get_triplets(embeddings,y_batch,max_neg,max_trips,factors,debug=False)
#         if trips is None:
#             continue
#         anch,pos,neg=trips
#         # compute loss
#         loss_batch=triplet_loss(anch,pos,neg)
#         # loss_batch = masked_loss_function(yy, embeddings)
#         loss_values.append(loss_batch.detach().numpy())
#         # backprop
#         optimizer.zero_grad()
#         loss_batch.backward(retain_graph=True)
#         optimizer.step()
#         print("Batch size :",anch.shape[0],", Loss value :",loss_batch.detach().numpy())
#     loss_mean=np.mean(np.array(loss_values))
#     train_acc = get_acc(X_train, Y_train, X_train, Y_train, model, 5, model_name)
#     val_acc = get_acc(X_train, Y_train, X_val, Y_val, model, 5, model_name)
#     print("\tTrain Loss for this epoch :",loss_mean)
#     print("\tTrain Accuracy for this epoch:", train_acc)
#     print("\tValidation Accuracy for this epoch:", val_acc)




# ROC_Score(model, X_test, Y_test,X_train, Y_train, limited=False)
# score_for_each_drug = ROC(model, X_test, Y_test, X_train, Y_train, ("LRCN" + "BO_delete"), True)
# spec_recall, prec_recall = PR(model, X_test, Y_test, X_train, Y_train)
# print("recall at 95 spec: ", spec_recall)
# print("precision recall: ", prec_recall)