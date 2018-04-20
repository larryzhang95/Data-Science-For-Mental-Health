from feature_embeddings import getAudioSet
from __future__ import print_function
import os
import sys

from scipy.io import wavfile
import pandas as pd
import numpy as np
import six
import tensorflow as tf
import pickle
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

#Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
"""
Method(s) for building model pipelines on top of audio feature embeddings
"""


#Data Retrieval
def getAllData(dir):
    dataList = os.listdir(dir)
    print(dataList)
    print(len(dataList))
    embeddingsList = []
    for wavFile in dataList:
        audio = os.path.join(dir,wavFile)
        seq, ppc_batch = getAudioSet(audio,pca_param_np,None,vggish_cpkt)
        embeddingsList.append(ppc_batch)
    embeddingsList = np.array(embeddingsList)
    return embeddingsList

#Pickle Functions For Saving and Retrieving Data:
def saveToPickle(vec):
    pickle.dump(vec, open('bipolar_data.p','wb'))
def getFromPickle(file):
    data = pickle.load(open(file,'rb'))
    return data

#Flattening embeddings
def getMeanEmbedding(embedding):
    embedding = np.array(embedding)
    return embedding.mean(axis=0)
def getMinEmbedding(embedding):
    embedding = np.array(embedding)
    return embedding.min(axis=0)
def getMaxEmbedding(embedding):
    embedding = np.array(embedding)
    return embedding.max(axis=0)
def getVarEmbedding(embedding):
    embedding= np.array(embedding)
    return embedding.var(axis=0)

def getEmbedding(data_vec,flag):
    data_embedded = np.empty(len(data_vec),data_vec.shape[2])
    c = 0
    for embedding in data_vec:
        if flag == 0:
            e = getMeanEmbedding(embedding)
        if flag == 1:
            e = getMinEmbedding(embedding)
        if flag == 2:
            e = getMaxEmbedding(embedding)
        if flag == 3:
            e = getVarEmbedding(embedding)
        else:
            print("Incorrect Flag")
            sys.exit()
        d[c] = e
        c += 1
     return data_embedded


#Example Model(s)
def build_svm_model(data_embedded,label):
    X_train, X_test, y_train, y_test = train_test_split(data_embedded,label,test_size=0.5)
    print(len(X_train),len(y_train))
    print(len(X_test),len(y_test))
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    preds = clf.predict(X_test)
    print(y_test)
    print(preds)
    accuracy = np.mean(preds == y_test)
    print(accuracy)
    return accuracy

def nn_model():
    model = Sequential()
    model.add(Dense(10,input_dim=128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(2,activation='linear'))
    model.compile(loss='mse',metrics=['accuracy'], optimizer='adam')
    return model

def build_neural_work(data_embedded, label):
    X_train, X_test, y_train, y_test = train_test_split(data_embedded,label,test_size=0.5)
    tuned_parameters = [{"epochs":[10,100],
                             "batch_size":[32,64,128,256]}]
    model = KerasClassifier(build_fn = nn_model, verbose = 0)
    clf = GridSearchCV(estimator=model,param_grid=tuned_parameters)
    clf.fit(X_train,y_train)
    print("Best parameters set found on development set:")
    print()
    print(grid_result.best_params_)
    preds = grid_result.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(accuracy)
    return accuracy
