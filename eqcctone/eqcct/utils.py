from __future__ import print_function
from __future__ import division
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib
import tensorflow
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv
from tensorflow import keras
import time
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
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten, Reshape, multiply
from tensorflow.keras.layers import concatenate, GRU, Input, LSTM, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Add, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, SeparableConv1D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten, UpSampling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras import layers, models, optimizers
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, GlobalAveragePooling1D
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Conv2DTranspose, Bidirectional, GRU, LSTM, Input,Dense, SpatialDropout1D, Conv2D, MaxPooling2D, Flatten, Input, UpSampling2D, Dropout,Lambda, Average, concatenate, Activation, Add
import numpy as np
from tensorflow.keras.layers import Input,Dense, Add, UpSampling1D, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Input, UpSampling2D, Dropout,Lambda, Average, concatenate, Activation
from tensorflow.keras import optimizers, Model
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
from numpy.random import seed
import math
from tensorflow.keras.regularizers import l2
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False




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
drop_rate = 0.2
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
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    





def convF1(inpt, D1, fil_ord, Dr):

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
    
    

#def create_vit_classifier(inputs):
def create_cct_model(inputs):

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

def create_cct_modelP(inputs):

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


#def create_vit_classifier(inputs):
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


def mseed_predictor(input_dir='downloads_mseeds',
              input_model="sampleData&Model/EqT1D8pre_048.h5",
              stations_json= "station_list.json",
              output_dir="detections",
              P_threshold=0.1,
              S_threshold=0.1, 
              number_of_plots=10,
              normalization_mode='std',
              batch_size=500,              
              overlap = 0.3,
              gpuid=None,
              gpu_limit=None,
              overwrite=False): 
    
    """ 
    
    To perform fast detection directly on mseed data.
    
    Parameters
    ----------
    input_dir: str
        Directory name containing hdf5 and csv files-preprocessed data.
            
    input_model: str
        Path to a trained model.
            
    stations_json: str
        Path to a JSON file containing station information. 
           
    output_dir: str
        Output directory that will be generated.
            
    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.                
            
    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
            
    normalization_mode: str, default=std
        Mode of normalization for data preprocessing max maximum amplitude among three components std standard deviation.
             
    batch_size: int, default=500
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommanded.
             
    overlap: float, default=0.3
        If set the detection and picking are performed in overlapping windows.
             
    gpuid: int
        Id of GPU used for the prediction. If using CPU set to None.        
             
    gpu_limit: int
       Set the maximum percentage of memory usage for the GPU. 

    overwrite: Bolean, default=False
        Overwrite your results automatically.
           
    Returns
    --------        
    
    
    """  
        
 
    args = {
    "input_dir": input_dir,
    "input_model": input_model,
    "stations_json": stations_json,
    "output_dir": output_dir,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "normalization_mode": normalization_mode,
    "overlap": overlap,
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

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                        datefmt='%m-%d %H:%M') 

    class DummyFile(object):
        file = None
        def __init__(self, file, **kwargs):
            self.file = file
    
        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x, file=self.file)
    
    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = DummyFile(sys.stdout)
        yield
        sys.stdout = save_stdout
    
 
    # print('============================================================================')
    eqt_logger = logging.getLogger("EQCCT_Picker")
    eqt_logger.info(f"Running EQCCT_Picker")
            
    eqt_logger.info(f"*** Loading the model ...")

    # Model EQCCT
    inputs = layers.Input(shape=input_shape,name='input')

    featuresP = create_cct_model(inputs)
    featuresP = Reshape((6000,1))(featuresP)

    featuresS = create_cct_model(inputs)
    featuresS = Reshape((6000,1))(featuresS)

    logitp  = Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_P')(featuresP)
    logits  = Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_S')(featuresS)



    
    model = Model(inputs=[inputs], outputs=[logitp,logits])
    model.summary()

    sgd = optimizers.Adam()
    lamP = 0.4
    lamS = 0.6
    model.compile(optimizer=sgd,
                  loss=['binary_crossentropy','binary_crossentropy'],
                  loss_weights=[lamP,lamS],
                  metrics=['acc',f1,precision, recall])    
    
    model.load_weights(args['input_model'])

    eqt_logger.info(f"*** Loading is complete!")


    out_dir = os.path.join(os.getcwd(), str(args['output_dir']))
    if os.path.isdir(out_dir):
        # print('============================================================================')        
        # print(f' *** {out_dir} already exists!')
        eqt_logger.info(f"*** {out_dir} already exists!")
        if overwrite == True:
            inp = "y"
            eqt_logger.info(f"Overwriting your previous results")
            # print("Overwriting your previous results")
        else:
            inp = input(" --> Type (Yes or y) to create a new empty directory! This will erase your previous results so make a copy if you want them.")
        if inp.lower() == "yes" or inp.lower() == "y":
            shutil.rmtree(out_dir)  
            os.makedirs(out_dir) 
        else:
            print("Okay.")
            return
     
    if platform.system() == 'Windows':
        station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("\\")[-1] != ".DS_Store"];
    else:     
        station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"];
        

    station_list = sorted(set(station_list))
    
    data_track = dict()

    # print(f"######### There are files for {len(station_list)} stations in {args['input_dir']} directory. #########", flush=True)
    eqt_logger.info(f"There are files for {len(station_list)} stations in {args['input_dir']} directory.")
    for ct, st in enumerate(station_list):
    
        save_dir = os.path.join(out_dir, str(st)+'_outputs')
        save_figs = os.path.join(save_dir, 'figures') 
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_dir) 
        #if args['number_of_plots']:
        #    os.makedirs(save_figs)
            
        plt_n = 0            
        csvPr_gen = open(os.path.join(save_dir,'X_prediction_results.csv'), 'w')          
        predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        predict_writer.writerow(['file_name', 
                                 'network',
                                 'station',
                                 'instrument_type',
                                 'station_lat',
                                 'station_lon',
                                 'station_elv',
                                 'p_arrival_time',
                                 'p_probability',
                                 's_arrival_time',
                                 's_probability'])  
        csvPr_gen.flush()
        # print(f'========= Started working on {st}, {ct+1} out of {len(station_list)} ...', flush=True)
        eqt_logger.info(f"Started working on {st}, {ct+1} out of {len(station_list)} ...")
        

        start_Predicting = time.time()       
        if platform.system() == 'Windows':
            file_list = [join(st, ev) for ev in listdir(args["input_dir"]+"\\"+st) if ev.split("\\")[-1].split(".")[-1].lower() == "mseed"]; 
        else:
            file_list = [join(st, ev) for ev in listdir(args["input_dir"]+"/"+st) if ev.split("/")[-1].split(".")[-1].lower() == "mseed"]; 
        
        mon = [ev.split('__')[1]+'__'+ev.split('__')[2] for ev in file_list ];
        uni_list = list(set(mon))
        uni_list.sort()  
          
        time_slots, comp_types = [], []
        
        # print('============ Station {} has {} chunks of data.'.format(st, len(uni_list)), flush=True)      
        for _, month in enumerate(uni_list):
            eqt_logger.info(f"{month}")
            matching = [s for s in file_list if month in s]
            matching.sort()
            #print(matching)
            meta, time_slots, comp_types, data_set = _mseed2nparry(args, matching, time_slots, comp_types, st)

            params_pred = {'batch_size': args['batch_size'],
                           'norm_mode': args['normalization_mode']}  
                
            pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)

            predP,predS = model.predict_generator(pred_generator)
            
            #print(np.shape(predP))
            detection_memory = []
            prob_memory=[]
            for ix in range(len(predP)):
                Ppicks, Pprob =  _picker(args, predP[ix,:, 0])   
                Spicks, Sprob =  _picker(args, predS[ix,:, 0], 'S_threshold') 
                #print(np.shape(Ppicks))
                detection_memory,prob_memory=_output_writter_prediction(meta, csvPr_gen, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, ix,len(predP),len(predS))
        
                                                       
        end_Predicting = time.time() 
        data_track[st]=[time_slots, comp_types] 
        delta = (end_Predicting - start_Predicting) 
        hour = int(delta / 3600)
        delta -= hour * 3600
        minute = int(delta / 60)
        delta -= minute * 60
        seconds = delta     
                        
        dd = pd.read_csv(os.path.join(save_dir,'X_prediction_results.csv'))
        print(f'\n', flush=True)
        eqt_logger.info(f"Finished the prediction in: {hour} hours and {minute} minutes and {round(seconds, 2)} seconds.")
        eqt_logger.info(f'*** Detected: '+str(len(dd))+' events.')
        eqt_logger.info(f' *** Wrote the results into --> " ' + str(save_dir)+' "')
        # print(' *** Finished the prediction in: {} hours and {} minutes and {} seconds.'.format(hour, minute, round(seconds, 2)), flush=True)         
        # print(' *** Detected: '+str(len(dd))+' events.', flush=True)
        # print(' *** Wrote the results into --> " ' + str(save_dir)+' "', flush=True)
        
        with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
            the_file.write('================== PREDICTION FROM MSEED ===================='+'\n')               
            the_file.write('================== Overal Info =============================='+'\n')               
            the_file.write('date of report: '+str(datetime.now())+'\n')         
            the_file.write('input_model: '+str(args['input_model'])+'\n')
            the_file.write('input_dir: '+str(args['input_dir'])+'\n')  
            the_file.write('output_dir: '+str(save_dir)+'\n')  
            the_file.write('================== Prediction Parameters ====================='+'\n')  
            the_file.write('finished the prediction in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2))) 
            #the_file.write('loss_types: '+str(args['loss_types'])+'\n')
            #the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
            the_file.write('================== Other Parameters =========================='+'\n')            
            the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
            the_file.write('overlap: '+str(args['overlap'])+'\n')  
            the_file.write('batch_size: '+str(args['batch_size'])+'\n')                                 
            #the_file.write('detection_threshold: '+str(args['detection_threshold'])+'\n')            
            #the_file.write('number_of_plots: '+str(args['number_of_plots'])+'\n')                        
            the_file.write('gpuid: '+str(args['gpuid'])+'\n')
            the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')    
  
    with open('time_tracks.pkl', 'wb') as f:
        pickle.dump(data_track, f, pickle.HIGHEST_PROTOCOL)

       
        
def mseed_predictor_two(input_dir='downloads_mseeds',
              input_modelP="sampleData&Model/EqT1D8pre_048.h5",
              input_modelS="sampleData&Model/EqT1D8pre_048.h5",
              stations_json= "station_list.json",
              output_dir="detections",
              P_threshold=0.1,
              S_threshold=0.1, 
              number_of_plots=10,
              normalization_mode='std',
              batch_size=500,              
              overlap = 0.3,
              gpuid=None,
              gpu_limit=None,
              overwrite=False): 
    
    """ 
    
    To perform fast detection directly on mseed data.
    
    Parameters
    ----------
    input_dir: str
        Directory name containing hdf5 and csv files-preprocessed data.
            
    input_model: str
        Path to a trained model.
            
    stations_json: str
        Path to a JSON file containing station information. 
           
    output_dir: str
        Output directory that will be generated.
            
    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.                
            
    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
            
    normalization_mode: str, default=std
        Mode of normalization for data preprocessing max maximum amplitude among three components std standard deviation.
             
    batch_size: int, default=500
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommanded.
             
    overlap: float, default=0.3
        If set the detection and picking are performed in overlapping windows.
             
    gpuid: int
        Id of GPU used for the prediction. If using CPU set to None.        
             
    gpu_limit: int
       Set the maximum percentage of memory usage for the GPU. 

    overwrite: Bolean, default=False
        Overwrite your results automatically.
           
    Returns
    --------        
    
    
    """  
        
 
    args = {
    "input_dir": input_dir,
    "input_modelP": input_modelP,
    "input_modelS": input_modelS,
    "stations_json": stations_json,
    "output_dir": output_dir,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "normalization_mode": normalization_mode,
    "overlap": overlap,
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

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                        datefmt='%m-%d %H:%M') 

    class DummyFile(object):
        file = None
        def __init__(self, file, **kwargs):
            self.file = file
    
        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x, file=self.file)
    
    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = DummyFile(sys.stdout)
        yield
        sys.stdout = save_stdout
    
 
    # print('============================================================================')
    eqt_logger = logging.getLogger("EQCCT_Picker")
    eqt_logger.info(f"Running EQCCT_Picker")
            
    eqt_logger.info(f"*** Loading the model ...")

    # Model CCT
    inputs = layers.Input(shape=input_shape,name='input')

    #featuresP = create_cct_model(inputs)
    #featuresS = create_cct_model(inputs)


    #logitp  = Dense(6000 ,activation='sigmoid', kernel_initializer='he_normal',name='picker_P1')(featuresP)
    #logits  = Dense(6000 ,activation='sigmoid', kernel_initializer='he_normal',name='picker_S1')(featuresS)
    #logitp = Reshape((6000,1), name='picker_P')(logitp)
    #logits = Reshape((6000,1), name='picker_S')(logits)


    featuresP = create_cct_modelP(inputs)
    featuresP = Reshape((6000,1))(featuresP)

    featuresS = create_cct_modelS(inputs)
    featuresS = Reshape((6000,1))(featuresS)

    logitp  = Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_P')(featuresP)
    logits  = Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_S')(featuresS)



    modelP = Model(inputs=[inputs], outputs=[logitp])
    modelS = Model(inputs=[inputs], outputs=[logits])

    model = Model(inputs=[inputs], outputs=[logitp,logits])
    model.summary()

    sgd = optimizers.Adam()
    model.compile(optimizer=sgd,
                  loss=['binary_crossentropy','binary_crossentropy'],
                  metrics=['acc',f1,precision, recall])    
    
    modelP.load_weights(args['input_modelP'])
    modelS.load_weights(args['input_modelS'])


    eqt_logger.info(f"*** Loading is complete!")


    out_dir = os.path.join(os.getcwd(), str(args['output_dir']))
    if os.path.isdir(out_dir):
        # print('============================================================================')        
        # print(f' *** {out_dir} already exists!')
        eqt_logger.info(f"*** {out_dir} already exists!")
        if overwrite == True:
            inp = "y"
            eqt_logger.info(f"Overwriting your previous results")
            # print("Overwriting your previous results")
        else:
            inp = input(" --> Type (Yes or y) to create a new empty directory! This will erase your previous results so make a copy if you want them.")
        if inp.lower() == "yes" or inp.lower() == "y":
            shutil.rmtree(out_dir)  
            os.makedirs(out_dir) 
        else:
            print("Okay.")
            return
     
    if platform.system() == 'Windows':
        station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("\\")[-1] != ".DS_Store"];
    else:     
        station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"];
        

    station_list = sorted(set(station_list))
    
    data_track = dict()

    # print(f"######### There are files for {len(station_list)} stations in {args['input_dir']} directory. #########", flush=True)
    eqt_logger.info(f"There are files for {len(station_list)} stations in {args['input_dir']} directory.")
    for ct, st in enumerate(station_list):
    
        save_dir = os.path.join(out_dir, str(st)+'_outputs')
        save_figs = os.path.join(save_dir, 'figures') 
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_dir) 
        #if args['number_of_plots']:
        #    os.makedirs(save_figs)
            
        plt_n = 0            
        csvPr_gen = open(os.path.join(save_dir,'X_prediction_results.csv'), 'w')          
        predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        predict_writer.writerow(['file_name', 
                                 'network',
                                 'station',
                                 'instrument_type',
                                 'station_lat',
                                 'station_lon',
                                 'station_elv',
                                 'p_arrival_time',
                                 'p_probability',
                                 's_arrival_time',
                                 's_probability'])  
        csvPr_gen.flush()
        # print(f'========= Started working on {st}, {ct+1} out of {len(station_list)} ...', flush=True)
        eqt_logger.info(f"Started working on {st}, {ct+1} out of {len(station_list)} ...")
        

        start_Predicting = time.time()       
        if platform.system() == 'Windows':
            file_list = [join(st, ev) for ev in listdir(args["input_dir"]+"\\"+st) if ev.split("\\")[-1].split(".")[-1].lower() == "mseed"]; 
        else:
            file_list = [join(st, ev) for ev in listdir(args["input_dir"]+"/"+st) if ev.split("/")[-1].split(".")[-1].lower() == "mseed"]; 
        
        mon = [ev.split('__')[1]+'__'+ev.split('__')[2] for ev in file_list ];
        uni_list = list(set(mon))
        uni_list.sort()  
          
        time_slots, comp_types = [], []
        
        # print('============ Station {} has {} chunks of data.'.format(st, len(uni_list)), flush=True)      
        for _, month in enumerate(uni_list):
            eqt_logger.info(f"{month}")
            matching = [s for s in file_list if month in s]
            matching.sort()
            #print(matching)
            meta, time_slots, comp_types, data_set = _mseed2nparry(args, matching, time_slots, comp_types, st)

            params_pred = {'batch_size': args['batch_size'],
                           'norm_mode': args['normalization_mode']}  
                
            pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)

            predP,predS = model.predict_generator(pred_generator)
            
            #print(np.shape(predP))
            detection_memory = []
            prob_memory=[]
            for ix in range(len(predP)):
                Ppicks, Pprob =  _picker(args, predP[ix,:, 0])   
                Spicks, Sprob =  _picker(args, predS[ix,:, 0]) 
                #print(np.shape(Ppicks))
                detection_memory,prob_memory=_output_writter_prediction(meta, csvPr_gen, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, ix,len(predP),len(predS))
        
                                                       
        end_Predicting = time.time() 
        data_track[st]=[time_slots, comp_types] 
        delta = (end_Predicting - start_Predicting) 
        hour = int(delta / 3600)
        delta -= hour * 3600
        minute = int(delta / 60)
        delta -= minute * 60
        seconds = delta     
                        
        dd = pd.read_csv(os.path.join(save_dir,'X_prediction_results.csv'))
        print(f'\n', flush=True)
        eqt_logger.info(f"Finished the prediction in: {hour} hours and {minute} minutes and {round(seconds, 2)} seconds.")
        eqt_logger.info(f'*** Detected: '+str(len(dd))+' events.')
        eqt_logger.info(f' *** Wrote the results into --> " ' + str(save_dir)+' "')
        # print(' *** Finished the prediction in: {} hours and {} minutes and {} seconds.'.format(hour, minute, round(seconds, 2)), flush=True)         
        # print(' *** Detected: '+str(len(dd))+' events.', flush=True)
        # print(' *** Wrote the results into --> " ' + str(save_dir)+' "', flush=True)
        
        with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
            the_file.write('================== PREDICTION FROM MSEED ===================='+'\n')               
            the_file.write('================== Overal Info =============================='+'\n')               
            the_file.write('date of report: '+str(datetime.now())+'\n')         
            the_file.write('input_dir: '+str(args['input_dir'])+'\n')  
            the_file.write('output_dir: '+str(save_dir)+'\n')  
            the_file.write('================== Prediction Parameters ====================='+'\n')  
            the_file.write('finished the prediction in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2))) 
            #the_file.write('loss_types: '+str(args['loss_types'])+'\n')
            #the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
            the_file.write('================== Other Parameters =========================='+'\n')            
            the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
            the_file.write('overlap: '+str(args['overlap'])+'\n')  
            the_file.write('batch_size: '+str(args['batch_size'])+'\n')                                 
            #the_file.write('detection_threshold: '+str(args['detection_threshold'])+'\n')            
            #the_file.write('number_of_plots: '+str(args['number_of_plots'])+'\n')                        
            the_file.write('gpuid: '+str(args['gpuid'])+'\n')
            the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')    
  
    with open('time_tracks.pkl', 'wb') as f:
        pickle.dump(data_track, f, pickle.HIGHEST_PROTOCOL)


def _mseed2nparry(args, matching, time_slots, comp_types, st_name):
    ' read miniseed files and from a list of string names and returns 3 dictionaries of numpy arrays, meta data, and time slice info'
    
    json_file = open(args['stations_json'])
    stations_ = json.load(json_file)
    
    st = obspy.core.Stream()
    tsw = False
    for m in matching: 
        #print(m)
        temp_st = read(os.path.join(str(args['input_dir']), m),debug_headers=True)
        if tsw == False and temp_st:
            tsw = True
            for tr in temp_st:
                time_slots.append((tr.stats.starttime, tr.stats.endtime))
        try:
            temp_st.merge(fill_value=0)                     
        except Exception:
            temp_st =_resampling(temp_st)
            temp_st.merge(fill_value=0) 
        temp_st.detrend('demean')
        st += temp_st
        
    #print(st)          
    st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
    st.taper(max_percentage=0.001, type='cosine', max_length=2)
    if len([tr for tr in st if tr.stats.sampling_rate != 100.0]) != 0:
        try:
            st.interpolate(100, method="linear")
        except Exception:
            st=_resampling(st)            
                    
    st.trim(min([tr.stats.starttime for tr in st]), max([tr.stats.endtime for tr in st]), pad=True, fill_value=0)

    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime  

    meta = {"start_time":start_time,
            "end_time": end_time,
            "trace_name":m
             } 
                
    chanL = [tr.stats.channel[-1] for tr in st]
    comp_types.append(len(chanL))
    tim_shift = int(60-(args['overlap']*60))
    next_slice = start_time+60  
    
    data_set={}
                
    sl = 0; st_times = []      
    while next_slice <= end_time:
        npz_data = np.zeros([6000, 3]) 
        st_times.append(str(start_time).replace('T', ' ').replace('Z', ''))
        w = st.slice(start_time, next_slice) 
        if 'Z' in chanL:
            npz_data[:,2] = w[chanL.index('Z')].data[:6000]
        if ('E' in chanL) or ('1' in chanL):    
            try: 
                npz_data[:,0] = w[chanL.index('E')].data[:6000]
            except Exception:
                npz_data[:,0] = w[chanL.index('1')].data[:6000]
        if ('N' in chanL) or ('2' in chanL):        
            try: 
                npz_data[:,1] = w[chanL.index('N')].data[:6000]
            except Exception:
                npz_data[:,1] = w[chanL.index('2')].data[:6000]
                
        data_set.update( {str(start_time).replace('T', ' ').replace('Z', '') : npz_data})
   
        start_time = start_time+tim_shift
        next_slice = next_slice+tim_shift 
        sl += 1
                         
    meta["trace_start_time"] = st_times
    #print(stations_)
    ''' 
    try:
        meta["receiver_code"]=st[0].stats.station
        meta["instrument_type"]=st[0].stats.channel[:2]
        meta["network_code"]=stations_[st[0].stats.station]['network']
        meta["receiver_latitude"]=stations_[st[0].stats.station]['coords'][0]
        meta["receiver_longitude"]=stations_[st[0].stats.station]['coords'][1]
        meta["receiver_elevation_m"]=stations_[st[0].stats.station]['coords'][2]  
    except Exception:
        meta["receiver_code"]=st_name
        meta["instrument_type"]=stations_[st_name]['channels'][0][:2]
        meta["network_code"]=stations_[st_name]['network']
        meta["receiver_latitude"]=stations_[st_name]['coords'][0]
        meta["receiver_longitude"]=stations_[st_name]['coords'][1]
        meta["receiver_elevation_m"]=stations_[st_name]['coords'][2] 
    '''
    try:
        meta["receiver_code"]=st[0].stats.station
        meta["instrument_type"]=0
        meta["network_code"]=0
        meta["receiver_latitude"]=0
        meta["receiver_longitude"]=0
        meta["receiver_elevation_m"]=s0
    except Exception:
        meta["receiver_code"]=st_name
        meta["instrument_type"]=0
        meta["network_code"]=0
        meta["receiver_latitude"]=0
        meta["receiver_longitude"]=0
        meta["receiver_elevation_m"]=0
                    
    return meta, time_slots, comp_types, data_set



class PreLoadGeneratorTest(tensorflow.keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. Pre-load version.
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32.
        Batch size.
            
    n_channels: int, default=3.
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'                
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'picker_P': y2, 'picker_S': y3}: outputs including two separate numpy arrays as labels for P, and S respectively.
    
    
    """  
    
    def __init__(self, 
                 list_IDs, 
                 inp_data, 
                 batch_size=32, 
                 norm_mode = 'std',
                 **kwargs):
       
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.inp_data = inp_data        
        self.on_epoch_end()
        self.norm_mode = norm_mode
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        try:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        except ZeroDivisionError:
            print("Your data duration in mseed file is too short! Try either longer files or reducing batch_size. ")
        
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def _normalize(self, data, mode = 'max'):          
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
        X = np.zeros((self.batch_size, 6000, 3))           
        # Generate data
        for i, ID in enumerate(list_IDs_temp):            
            data = self.inp_data[ID]
            data = self._normalize(data, self.norm_mode)                            
            X[i, :, :] = data                                                           
                           
        return X      
    

def _output_writter_prediction(meta, csvPr, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, idx, cq, cqq):

    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    predict_writer: obj
        For writing out the detection/picking results in the CSV file. 
       
    csvPr: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    station_lat = meta["receiver_latitude"]
    station_lon = meta["receiver_longitude"]
    station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    station_name = "{:<4}".format(station_name)
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
  
    
    p_time = []
    p_prob = []
    PdateTime = []
    if Ppicks[0]!=None: 
#for iP in range(len(Ppicks)):
#if Ppicks[iP]!=None: 
        p_time.append(start_time+timedelta(seconds= Ppicks[0]/100))
        p_prob.append(Pprob[0])
        PdateTime.append(_date_convertor(start_time+timedelta(seconds= Ppicks[0]/100)))
        detection_memory.append(p_time) 
        prob_memory.append(p_prob)  
    else:          
        p_time.append(None)
        p_prob.append(None)
        PdateTime.append(None)
   
  



            

    s_time = []
    s_prob = []    
    SdateTime=[]
    if Spicks[0]!=None: 
#for iS in range(len(Spicks)):
#if Spicks[iS]!=None: 
        s_time.append(start_time+timedelta(seconds= Spicks[0]/100))
        s_prob.append(Sprob[0])
        SdateTime.append(_date_convertor(start_time+timedelta(seconds= Spicks[0]/100)))
    else:
        s_time.append(None)
        s_prob.append(None)
        SdateTime.append(None)

    SdateTime = np.array(SdateTime)
    s_prob = np.array(s_prob)
    
    p_prob = np.array(p_prob)
    PdateTime = np.array(PdateTime)
        
    predict_writer.writerow([meta["trace_name"], 
                     network_name,
                     station_name, 
                     instrument_type,
                     station_lat, 
                     station_lon,
                     station_elv,
                     PdateTime[0], 
                     p_prob[0],
                     SdateTime[0], 
                     s_prob[0]
                     ]) 



    csvPr.flush()                


    return detection_memory,prob_memory
            


def _get_snr(data, pat, window=200):
    
    """ 
    
    Estimates SNR.
    
    Parameters
    ----------
    data : numpy array
        3 component data.    
        
    pat: positive integer
        Sample point where a specific phase arrives. 
        
    window: positive integer, default=200
        The length of the window for calculating the SNR (in the sample).         
        
    Returns
   --------   
    snr : {float, None}
       Estimated SNR in db. 
       
        
    """      
       
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)    
        except Exception:
            pass
    return snr 



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
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
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





def _picker(args, yh3, thr_type='P_threshold'):
    """ 
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
         probability. 

    Returns
    --------    
    Ppickall: Pick.
    Pproball: Pick Probability.                           
                
    """
    P_PICKall=[]
    Ppickall=[]
    Pproball = []
    perrorall=[]

    sP_arr = _detect_peaks(yh3, mph=args[thr_type], mpd=1)

    P_PICKS = []
    pick_errors = []
    if len(sP_arr) > 0:
        P_uncertainty = None  

        for pick in range(len(sP_arr)):        
            sauto = sP_arr[pick]


            if sauto: 
                P_prob = np.round(yh3[int(sauto)], 3) 
                P_PICKS.append([sauto,P_prob, P_uncertainty]) 

    so=[]
    si=[]
    P_PICKS = np.array(P_PICKS)
    P_PICKall.append(P_PICKS)
    for ij in P_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        Ppickall.append((swave))
        Pproball.append((np.max(so)))
    except:
        Ppickall.append(None)
        Pproball.append(None)

    #print(np.shape(Ppickall))
    #Ppickall = np.array(Ppickall)
    #Pproball = np.array(Pproball)
    
    return Ppickall, Pproball


def _resampling(st):
    'perform resampling on Obspy stream objects'
    
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr) 
    return st 



def _normalize(data, mode = 'max'):  
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """  
       
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
    



