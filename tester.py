# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:16:01 2018

@author: anish
"""

import numpy as np
import pickle
from PIL import Image
from lr import sigmoid
from lr import predict


with open("parameters_lr.txt","rb") as fp:
    parameters=pickle.load(fp)



img=Image.open("test5.jpg")
img=img.resize((64,64))
X=np.asarray(img)
X=X.reshape(X.shape[0]*X.shape[1]*X.shape[2],1)
X=X/255

print(predict(parameters["w"],parameters["b"],X))