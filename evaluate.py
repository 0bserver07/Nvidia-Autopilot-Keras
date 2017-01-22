from __future__ import absolute_import
from __future__ import print_function
import os


os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'


import keras
import keras.models as models

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.metrics as metrics

from keras.callbacks import ModelCheckpoint

import cv2
import numpy as np
import json

# from Dan Does Data VLOG
import math
import h5py
import glob
from tqdm import tqdm
import scipy
from scipy import misc

import matplotlib.pyplot as plt
plt.ion()


ndata = 0
imgsize = 128
# frame size
nrows = 128
ncols = 128



imgs = np.load('data/imgs_128.npz')['arr_0']
targets = np.load('data/targets_128.npz')['arr_0']

idx = np.arange(0,imgs.shape[0])
idx = np.random.permutation(idx)
imgs = imgs[idx,:,:,:]
targets = targets[idx,:]




# load the model:
model = Sequential()
with open('autopilot_basic_model.json') as model_file:
    model = models.model_from_json(model_file.read())


# load weights
model.load_weights("weights/model_basic_weight.hdf5")


adam = Adam(lr=0.0001)
model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse','accuracy'])




preds = model.predict(imgs)
steer_preds = (preds.reshape([-1])+1)/2.
plt.plot(np.array([targets[:,0],steer_preds.reshape(len(steer_preds))]).T,'.')



# Animation!
def get_point(s,start=0,end=128,height= 16):
    X = int(s*(end-start))
    if X < start:
        X = start
    if X > end:
        X = end
    return (X,height)

val_idx = 0

from PIL import Image, ImageDraw

import matplotlib.animation as animation
figure = plt.figure()
imageplot = plt.imshow(np.zeros((128, 128, 3), dtype=np.uint8))

# needs fixing!
# def next_frame(i):
# im = Image.fromarray(np.array(imgs[val_idx+i].transpose(1,2,0),dtype=np.uint8))
# p = get_point(1-steer_preds[i])
# t = get_point(1-targets[i+val_idx,0])
# draw = ImageDraw.Draw(im) 
# draw.line((64,128, p,p),fill=(255,0,0,128))
# draw.line((64,128, t,t),fill=(0,255,0,128))
# imageplot.set_array(im)
# 	return imageplot,
# animate = animation.FuncAnimation(figure, next_frame, frames=range(0,len(imgs)), interval=25, blit=False)
# plt.show()