from __future__ import print_function
from __future__ import division


import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

import os
os.environ['KERAS_BACKEND']='tensorflow'

from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter, lfilter, lfilter_zi
import matplotlib
from tensorflow import keras

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow
matplotlib.use('agg')
import pandas as pd
import math
import csv
import time
import random
from os import listdir
import platform
import shutil
from tqdm import tqdm
from datetime import datetime, timedelta
import contextlib
import sys
import warnings
from scipy import signal
from matplotlib.lines import Line2D
from obspy import read
from os.path import join
import json
import pickle
import faulthandler; faulthandler.enable()
import obspy
import logging
from obspy.signal.trigger import trigger_onset
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint, LearningRateScheduler
from sklearn.utils import class_weight
from numpy.random import seed
from tensorflow.keras.regularizers import l2
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def picker(args, yh3, yh3_std, sst=None):
    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
     
        
    yh3 : 1D array
        S arrival probabilities. 
        
    yh3_std : 1D array
        S arrival standard deviations. 
        
    sst : {int, None}, default=None    
        S arrival time in sample.
        
   
    Returns
    --------    
    matches: dic
        Contains the information for the detected and picked event.            
        
    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
            
    pick_errors : dic                
        {detection statr-time:[ s_ground_truth - s_pick, S_ground_truth - S_pick]}
        
    yh3: 1D array             
        normalized S_probability                              
                
    """               
        


    s_PICKall=[]
    Spickall=[]
    Sproball = []
    serrorall=[]



    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)

    s_PICKS = []
    pick_errors = []
    if len(ss_arr) > 0:
        s_uncertainty = None  

        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]

            if  args['estimate_uncertainty'] and sauto:
                s_uncertainty = np.round(yh3_std[int(sauto)], 3)

            if sauto: 
                s_prob = np.round(yh3[int(sauto)], 3) 
                s_PICKS.append([sauto,s_prob, s_uncertainty]) 

    so=[]
    si=[]
    s_PICKS = np.array(s_PICKS)
    s_PICKall.append(s_PICKS)
    for ij in s_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        serrorall.append(int(sst- swave))  
        Spickall.append(int(swave))
        Sproball.append(int(np.max(so)))
    except:
        serrorall.append(None)
        Spickall.append(None)
        Sproball.append(None)


    #Spickall = np.array(Spickall)
    #serrorall = np.array(serrorall)  
    #Sproball = np.array(Sproball)
    
    return Spickall, serrorall, Sproball

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def generate_arrays_from_file(file_list, step):
    
    """ 
    
    Make a generator to generate list of trace names.
    
    Parameters
    ----------
    file_list : str
        A list of trace names.  
        
    step : int
        Batch size.  
        
    Returns
    --------  
    chunck : str
        A batch of trace names. 
        
    """     
    
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck   
class DataGeneratorTest(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 norm_mode = 'max'):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data    


    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            dataset = fl.get(str(ID))
            data = np.array(dataset['data'])              

          
            if self.norm_mode:                    
                data = self.normalize(data, self.norm_mode)  
                            
            X[i, :, :] = data                                       

        fl.close() 
                           
        return X

def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1(y_true, y_pred):
           
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))


import tensorflow as tf
def wbceEdit( y_true, y_pred) :
    
    ms = K.mean(K.square(y_true-y_pred)) 
    
    ssim = 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))
    
    return (ssim + ms)

w1 = 6000
w2 = 3
dros_rate = 0.2
stochastic_depth_rate = 0.1

positional_emb = False
conv_layers = 4
num_classes = 1
input_shape = (w1, w2)
num_classes = 1
input_shape = (6000, 3)
image_size = 6000  # We'll resize input images to this size
patch_size = 40  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size)
projection_dim = 40

num_heads = 4
transformer_units = [
    projection_dim,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded
    
# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, dros_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.dros_prob = dros_prop

    def call(self, x, training=None):
        if training:
            kees_prob = 1 - self.dros_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = kees_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / kees_prob) * random_tensor
        return x
    





def convF1(inpt, D1, fil_ord, Dr):
    '''
    encode = BatchNormalization()(inpt)    
    encode = Activation(tf.nn.gelu')(encode)
    encode = SpatialDropout1D(Dr)(encode, training=True)
    encode  = Conv1D(D1,  fil_ord, strides =(1), padding='same')(encode)
    
     
    encode  = Conv1D(D1,  fil_ord, strides =(1), padding='same')(inpt)
    encode = BatchNormalization()(encode)    
    encode = Activation(tf.nn.gelu')(encode)
    encode = Dropout(Dr)(encode)
    '''
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #filters = inpt._keras_shape[channel_axis]
    filters = int(inpt.shape[-1])
    
    #infx = Activation(tf.nn.gelu')(inpt)
    pre = Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inpt)
    pre = BatchNormalization()(pre)    
    pre = Activation(tf.nn.gelu)(pre)
    
    #shared_conv = Conv1D(D1,  fil_ord, strides =(1), padding='same')
    
    inf  = Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(pre)
    inf = BatchNormalization()(inf)    
    inf = Activation(tf.nn.gelu)(inf)
    inf = Add()([inf,inpt])
    
    inf1  = Conv1D(D1,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inf)
    inf1 = BatchNormalization()(inf1)  
    inf1 = Activation(tf.nn.gelu)(inf1)    
    encode = Dropout(Dr)(inf1)

    return encode

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        #x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded

def create_cct_modelP(inputs):

    inputs1 = convF1(inputs,   10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = layers.Reshape((6000,1,40))(inputs1)

    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        #encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        #attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def create_cct_modelS(inputs):

    inputs1 = convF1(inputs,   10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation

def test_swave(input_hdf5=None,
           input_testset=None,
           input_model=None,
           output_name=None,
           S_threshold=0.1, 
           number_of_plots=100,
           estimate_uncertainty=True, 
           number_of_sampling=5,
           input_dimention=(6000, 3),
           normalization_mode='std',
           mode='generator',
           batch_size=500,
           gpuid=None,
           gpu_limit=None):

    """
    
    Applies a trained model to a windowed waveform to perform both detection and picking at the same time.  


    Parameters
    ----------
    input_hdf5: str, default=None
        Path to an hdf5 file containing only one class of "data" with NumPy arrays containing 3 component waveforms each 1 min long.

    input_testset: npy, default=None
        Path to a NumPy file (automaticaly generated by the trainer) containing a list of trace names.        

    input_model: str, default=None
        Path to a trained model.
        
    output_dir: str, default=None
        Output directory that will be generated. 
        
    output_probabilities: bool, default=False
        If True, it will output probabilities and estimated uncertainties for each trace into an HDF file. 


    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
               
    number_of_plots: float, default=10
        The number of plots for detected events outputed for each station data.
        
    estimate_uncertainty: bool, default=False
        If True uncertainties in the output probabilities will be estimated.  
        
    number_of_sampling: int, default=5
        Number of sampling for the uncertainty estimation. 

        
    input_dimention: tuple, default=(6000, 3)
        Loss types for detection S picking.          

    normalization_mode: str, default='std' 
        Mode of normalization for data preprocessing, 'max', maximum amplitude among three components, 'std', standard deviation.

    mode: str, default='generator'
        Mode of running. 'pre_load_generator' or 'generator'.
                      
    batch_size: int, default=500 
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommanded.

    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None.
         
    gpu_limit: int, default=None
        Set the maximum percentage of memory usage for the GPU.
        
        
    """ 
              
         
    args = {
    "input_hdf5": input_hdf5,
    "input_testset": input_testset,
    "input_model": input_model,
    "output_name": output_name,
    "S_threshold": S_threshold,
    "number_of_plots": number_of_plots,
    "estimate_uncertainty": estimate_uncertainty,
    "number_of_sampling": number_of_sampling,
    "input_dimention": input_dimention,
    "normalization_mode": normalization_mode,
    "mode": mode,
    "batch_size": batch_size,
    "gpuid": gpuid,
    "gpu_limit": gpu_limit
    }  

    
    if args['gpuid']:           
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args['gpuid'])
        tf.Session(config=tf.ConfigProto(log_device_placement=True))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
        K.tensorflow_backend.set_session(tf.Session(config=config))
    
    save_dir = os.path.join(os.getcwd(), str(args['output_name'])+'_outputs')
    save_figs = os.path.join(save_dir, 'figures')
 
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)  
    os.makedirs(save_figs) 
 
    test = np.load(args['input_testset'])
    
    #test = test[0:1000]
    print('Loading the model ...', flush=True) 
     # Model CCT
    inputs = layers.Input(shape=input_shape,name='input')
    
    featuresP = create_cct_modelS(inputs)
    featuresP = Reshape((6000,1))(featuresP)

    logitp  = Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_S')(featuresP)
    modelP = Model(inputs=[inputs], outputs=[logitp])
    model = Model(inputs=[inputs], outputs=[logitp])   

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc',f1,precision, recall])          
    
    model.load_weights(args['input_model'])
    
    ''' 
    model = load_model(args['input_model'], custom_objects={'f1': f1, 'precision': precision, 
                                                            'recall':recall,
                                                            'ConvCapsuleLayer':ConvCapsuleLayer},compile=False)
                
    model.compile(loss = args['loss_types'],
                  loss_weights =  args['loss_weights'],           
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
    '''
    print('Loading is complete!', flush=True)  
    print('Testing ...', flush=True)    
    print('Writting results into: " ' + str(args['output_name'])+'_outputs'+' "', flush=True)
    
    start_training = time.time()          

    csvTst = open(os.path.join(save_dir,'X_test_results.csv'), 'w')          
    test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    test_writer.writerow([
                          
                          
                          's_arrival_sample', 
                          

                          
                          's_pick',
                          's_probability',
                          's_error'
                          ])  
    csvTst.flush()        
        
    plt_n = 0
    list_generator = generate_arrays_from_file(test, args['batch_size']) 
    pred_SS_mean_all=[]
    pred_SS_std_all=[] 
    sstall =[]
    
    pbar_test = tqdm(total= int(np.ceil(len(test)/args['batch_size'])))            
    for _ in range(int(np.ceil(len(test) / args['batch_size']))):
        pbar_test.update()
        new_list = next(list_generator)

        if args['mode'].lower() == 'pre_load_generator':                
            params_test = {'dim': args['input_dimention'][0],
                           'batch_size': len(new_list),
                           'n_channels': args['input_dimention'][-1],
                           'norm_mode': args['normalization_mode']}  
            test_set={}


        
        else:       
            params_test = {'file_name': str(args['input_hdf5']), 
                           'dim': args['input_dimention'][0],
                           'batch_size': len(new_list),
                           'n_channels': args['input_dimention'][-1],
                           'norm_mode': args['normalization_mode']}     
    
            test_generator = DataGeneratorTest(new_list, **params_test)
            
            if args['estimate_uncertainty']:

                pred_SS = []          
                for mc in range(args['number_of_sampling']):
                    preSS = model.predict_generator(generator=test_generator)
                    pred_SS.append(preSS)
                
    
                
                pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_SS_mean = pred_SS.mean(axis=0)
                pred_SS_std = pred_SS.std(axis=0) 

            else:          
                pred_SS_mean = model.predict_generator(generator=test_generator)
                pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1]) 
                
                pred_SS_std = np.zeros((pred_SS_mean.shape))   
                
   
            test_set={}
            fl = h5py.File(args['input_hdf5'], 'r')
            for ID in new_list:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get(str(ID))
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get(str(ID))
                test_set.update( {str(ID) : dataset})                 
            
            for ts in range(pred_SS_mean.shape[0]): 
                evi =  new_list[ts] 
                dataset = test_set[evi]  
                
                    
                try:
                    sst = int(dataset.attrs['s_arrival_sample']);
                except Exception:     
                    sst = None
                
                
                Spick, serror, Sprob =  picker(args, pred_SS_mean[ts], pred_SS_std[ts], sst) 
                
                _output_writter_test(args, dataset, evi, test_writer, csvTst, Spick, serror, Sprob)
                        
                pred_SS_mean_all.append(pred_SS_mean[ts])
                pred_SS_std_all.append(pred_SS_std[ts])
                sstall.append(sst)

    
    np.save('pred_SS_mean_all.npy',pred_SS_mean_all)
    np.save('pred_SS_std_all.npy',pred_SS_std_all)
    np.save('sall.npy',sstall)
    
    end_training = time.time()  
    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta     
                    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')            
        the_file.write('input_testset: '+str(args['input_testset'])+'\n')
        the_file.write('input_model: '+str(args['input_model'])+'\n')
        the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')  
        the_file.write('================== Testing Parameters ======================='+'\n')  
        the_file.write('mode: '+str(args['mode'])+'\n')  
        the_file.write('finished the test in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2))) 

        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('total number of tests '+str(len(test))+'\n')
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('================== Other Parameters ========================='+'\n')            
        the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
        the_file.write('estimate uncertainty: '+str(args['estimate_uncertainty'])+'\n')
        the_file.write('number of Monte Carlo sampling: '+str(args['number_of_sampling'])+'\n')                       
        the_file.write('S_threshold: '+str(args['S_threshold'])+'\n')

    
    
def _output_writter_test(args, 
                        dataset, 
                        evi, 
                        output_writer, 
                        csvfile, 
                        Spick,
                        serror,
                        Sprob
                        ):
    
    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.    
 
    dataset: hdf5 obj
        Dataset object of the trace.

    evi: str
        Trace name.    
              
    output_writer: obj
        For writing out the detection/picking results in the CSV file.
        
    csvfile: obj
        For writing out the detection/picking results in the CSV file.   
             
        
    Returns
    --------  
    X_test_results.csv  
    
        
    """        
    
    

    
    if evi.split('_')[-1] == 'EV':                                     

        s_arrival_sample = dataset.attrs['s_arrival_sample'] 

                   
    elif evi.split('_')[-1] == 'NO':               
        #network_code = dataset.attrs['network_code']
        source_id = None
        source_distance_km = None 
        snr_db = None
        #trace_name = dataset.attrs['trace_name'] 
        #trace_category = dataset.attrs['trace_category']            
        trace_start_time = None
        source_magnitude = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        #receiver_type = dataset.attrs['receiver_type'] 
      
    #print(Spick[0])

    output_writer.writerow([ 

                            s_arrival_sample, 
                             

            
                            Spick[0], 
                            Sprob[0],
                            serror[0],
                            
                            ]) 
    
    csvfile.flush()
