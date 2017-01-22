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


# checkpoint
filepath="weights/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


adam = Adam(lr=0.0001)
model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse','accuracy'])

nb_epoch = 25
batch_size = 64

model.fit(imgs, targets[:,0]*2-1, callbacks=callbacks_list,
	batch_size =batch_size, nb_epoch=nb_epoch, verbose=1,
	validation_split=0.1,shuffle=True)


model.save_weights('weights/model_basic_weight.hdf5')
