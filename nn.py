# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:39:36 2018

@author: anish
"""

from os import listdir
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def loaddata():
    mypath="D:\Download\cat-dataset\CAT_00"
    catfiles = [f for f in listdir(mypath)]
    for i in catfiles:
        img=Image.open(mypath+"\\"+i)
        img=img.resize((64,64))
        x=np.asarray(img)
        x=np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
        X_list.append(x)
        Y_list.append(1)
    mypath2="D:\Download\ObjectCategories\BACKGROUND_Google"
    notcatfiles = [g for g in listdir(mypath2)]
    for j in notcatfiles:
        img=Image.open(mypath2+"\\"+j)
        img=img.resize((64,64))
        x=np.asarray(img)
        if x.shape!=(64,64,3):
            continue
        x=np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
        X_list.append(x)
        Y_list.append(0)
#        with open("dataX.txt","wb") as fp:
#            pickle.dump(X,fp)
#        with open("dataY.txt","wb") as fp:
#            pickle.dump(Y,fp)

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def layer_sizes(X, Y):  
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_y)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.rand(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.rand(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}   
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
   
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache   

def compute_cost(A2, Y, parameters):    
    m = Y.shape[1] # number of example
    logprobs = np.dot(Y,np.log(A2).T)+np.dot(1-Y,np.log(1-A2).T)
    cost = (-1/m)*sum(logprobs)
    cost= np.squeeze(cost,axis=0) 
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}  
    return grads


def update_parameters(parameters, grads, learning_rate = 0.05):   
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]   
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]   
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000,print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]    
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs=[]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)        
        grads = backward_propagation(parameters, cache, X, Y)        
        parameters = update_parameters(parameters, grads)
        if i%100==0:
            costs.append(cost)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return costs,parameters

def predict(parameters, X):  
    A2, cache = forward_propagation(X, parameters)
    predictions=(A2>0.5)       
    return predictions

if __name__=="__main__":    
#    if path.isfile("dataX.txt"):
#        with open("dataX.txt","rb") as fp: 
#            X_list=pickle.load(fp)
#        with open("dataY.txt","rb") as fp:
#            Y_list=pickle.load(fp)
#    else:             
    X_list=[]
    Y_list=[]
    loaddata()
    c = list(zip(X_list, Y_list))
    random.shuffle(c)
    X_list, Y_list = zip(*c)
    X=np.array(X_list)
    X=X.reshape((X.shape[0],X.shape[1]))
    X=X.T
    Y=np.array(Y_list)
    Y=Y.reshape((1,Y.shape[0]))
    X=X/255
    X_trainset=X[:,0:1500]
    Y_trainset=Y[:,0:1500]
    X_testset=X[:,1500:]
    Y_testset=Y[:,1500:]
    n_h=4
    costs,parameters = nn_model(X_trainset, Y_trainset, n_h , num_iterations = 5000, print_cost=True)
    with open("parameters.txt","wb") as fp:
        pickle.dump(parameters,fp)
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Hidden Layers: %d"%n_h)
    plt.show()
    
    
    predictions_train = predict(parameters, X_trainset)
    print ('Accuracy for training set: %d' % float((np.dot(Y_trainset,predictions_train.T) + np.dot(1-Y_trainset,1-predictions_train.T))/float(Y_trainset.size)*100) + '%')
    
    predictions_test = predict(parameters, X_testset)
    print ('Accuracy for test set: %d' % float((np.dot(Y_testset,predictions_test.T) + np.dot(1-Y_testset,1-predictions_test.T))/float(Y_testset.size)*100) + '%')



