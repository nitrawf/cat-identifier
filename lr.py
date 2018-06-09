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

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def initialize(dim):
    w=np.zeros((dim,1),dtype=float)
    b=0
    return w,b

def propagate(w, b, X, Y):        
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)                                 
    cost = (-1/m)*np.sum(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))              
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)    
    cost = np.squeeze(cost)  
    grads = {"dw": dw,
             "db": db}    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):     
    costs = []      
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]      
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i%100==0:
            costs.append(cost)       
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))   
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}    
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)        
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0][i]<=0.75:
            Y_prediction[0][i]=0
        else:
            Y_prediction[0][i]=1   
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):  
    w, b = initialize(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)        
    w = parameters["w"]
    b = parameters["b"]    
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)   
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)) 
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}   
    return d  

       

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
    d = model(X_trainset, Y_trainset, X_testset, Y_testset, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    costs = np.squeeze(d['costs'])
    with open("parameters_lr.txt","wb") as fp:
        pickle.dump(d,fp)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
    