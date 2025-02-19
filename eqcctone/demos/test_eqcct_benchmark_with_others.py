#conda activate eqcct (see github page for envs)
#
#
# Requirement: TXED (github txed), version: TXED_20231111.h5
# 
# Import python library

from __future__ import print_function
from __future__ import division
import os
import tensorflow as tf
os.environ['KERAS_BACKEND']='tensorflow'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()


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
from tensorflow.keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.utils import class_weight
from numpy.random import seed
from tensorflow.keras.regularizers import l2
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from eqcct.dependence_p import *
from eqcct.dependence_s import *

## Hyperparameter setup of EQCCT
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
image_size = 6000  
patch_size = 40  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size)
projection_dim = 40

num_heads = 4
transformer_units = [
    projection_dim,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4


##########################################################################################
## EQCCT for P-wave arrival prediction
##########################################################################################

## Change TXED path below
test_pwave(input_hdf5= '/Users/chenyk/DATALIB/TXED/TXED_20231111.h5', # here you need to change the input path of your own data
       input_testset='../data/tests/signalid_random_1.49w.npy',
       #input_model='/home/g202321530/Yang/Yangtze/EQ_classification/Texas_1/txed-main/demos/EQCCT_P/test_trainer_outputs/models/test_trainer_001.h5',
       input_model='../ModelPS/test_trainer_024.h5',
       output_name='test_pwave',
       detection_threshold=0.1,                
       P_threshold=0.1,
       number_of_plots=3,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=1024,
       gpuid=None,
       gpu_limit=None)
##########################################################################################
## EQCCT for S-wave arrival prediction
##########################################################################################
test_swave(input_hdf5='/Users/chenyk/DATALIB/TXED/TXED_20231111.h5', # here you need to change the input path of your own data
       input_testset='../data/tests/signalid_random_1.49w.npy',
       #input_model='/home/g202321530/Yang/Yangtze/EQ_classification/Texas_1/txed-main/demos/EQCCT_S/test_trainer_S_outputs/models/test_trainer_S_001.h5',
       input_model='../ModelPS/test_trainer_021.h5',
       output_name='test_swave',
       S_threshold=0.1, 
       number_of_plots=3,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=1024,
       gpuid=None,
       gpu_limit=None)

##########################################################################################
### For refined model from Yang Cui
##########################################################################################
## EQCCT for P-wave arrival prediction
test_pwave(input_hdf5= '/Users/chenyk/DATALIB/TXED/TXED_20231111.h5', # here you need to change the input path of your own data
       input_testset='../data/tests/signalid_random_1.49w.npy',
       #input_model='/home/g202321530/Yang/Yangtze/EQ_classification/Texas_1/txed-main/demos/EQCCT_P/test_trainer_outputs/models/test_trainer_001.h5',
       input_model='../ModelPS/test_trainer_EQCCT_P_retrain_yangcui.h5',
       output_name='test_pwave_retrain',
       detection_threshold=0.1,                
       P_threshold=0.1,
       number_of_plots=3,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=1024,
       gpuid=None,
       gpu_limit=None)
##########################################################################################
## EQCCT for S-wave arrival prediction
test_swave(input_hdf5='/Users/chenyk/DATALIB/TXED/TXED_20231111.h5', # here you need to change the input path of your own data
       input_testset='../data/tests/signalid_random_1.49w.npy',
       input_model='../ModelPS/test_trainer_EQCCT_S_retrain_yangcui.h5',
       #input_model='./model/test_trainer_021.h5',
       output_name='test_swave_retrain',
       S_threshold=0.1, 
       number_of_plots=3,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=1024,
       gpuid=None,
       gpu_limit=None)
##########################################################################################

##########################################################################################
### For Yang Cui's model (HANet, hybrid attention network) 
# Cui, Y., M. Bai, J. Wu, and Y. Chen. "Earthquake signal detection using a multi-scale feature fusion network with hybrid attention mechanism." Geophysical Journal International (2024): ggae423.
##########################################################################################
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter, lfilter, lfilter_zi
import tensorflow as tf

seed = 202404
np.random.seed(seed)
tf.random.set_seed(seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

import os


# Make a new dir to save the picking error
directory_path2 = "figs"

# Create the directory, including any necessary parent directories
os.makedirs(directory_path2, exist_ok=True)

  
## Define a band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y


## Deine the evaluation metrics
def evaluate_picking(pred_array, label_array):
    pred_indx_array = []
    label_indx_array = []
    for i in range (pred_array.shape[0]-1):
        pred_indx = np.argmax(pred_array[i, :])
        # label_indx = np.argmax(label_array[i, :])
        label_indx = label_array[i]
        pred_indx_array.append(pred_indx)
        label_indx_array.append(label_indx)

    pred_indx_array = np.array(pred_indx_array)
    label_indx_array = np.array(label_indx_array)

    # calculate the error index
    err_indx_array = label_indx_array- pred_indx_array

    return err_indx_array

def cal_mae_std(input_array, m):
    # from sample to seconds
    input_array = input_array/100
    # filter those outliers
    input_array[(input_array > m) | (input_array < -m)] = 0

    # calculate the evaluation metrics
    mae = np.mean(np.abs(input_array - np.mean(input_array)))
    std = np.std(input_array)

    # calculate the precision of picking results (within ±1 s)
    precision = (np.sum(np.abs(input_array) <= 1) / len(input_array)) * 100
    
    return input_array, mae, std, precision 


## Load data and calculate the arrivals
# Load the TXED dataset
# please down load the TXED from: https://drive.google.com/drive/folders/1WXVB8ytNB4bOaZ97oq6OmMRyAEg95trp?usp=sharing
f = h5py.File("/Users/chenyk/DATALIB/TXED/TXED_20231111.h5", 'r')
# Load randomly selected random id
event_id = np.load('../data/tests/signalid_random_1.49w.npy')


# obtain the P- and S-wave arrivals
P_arrival_list = []
S_arrival_list = []
print('-----------arrival time calculation begin-------------------')
for key in event_id:
    if key in f:   
        dataset = f.get(key)
        P_arrival_list.append(int(dataset.attrs['p_arrival_sample']))
        S_arrival_list.append(int(dataset.attrs['s_arrival_sample']))
P_arrival_list = np.array(P_arrival_list)
S_arrival_list = np.array(S_arrival_list)
P_phase_label = P_arrival_list
S_phase_label = S_arrival_list

print(P_arrival_list.shape, P_arrival_list)
print(S_arrival_list.shape, S_arrival_list)
print('-----------arrival time calculation end-------------------')

# band-pass and normalization of the 3-C waveforms
signal_list = []
print('-----------signal format convert begin-------------------')
for key in event_id:
    if key in f:   
        dataset = f.get(key)
        datas = dataset['data']
        datas = np.array(datas)
        datas_0 = butter_bandpass_filter_zi(datas[:,0], 1, 45, 100, order=3)
        datas_1 = butter_bandpass_filter_zi(datas[:,1], 1, 45, 100, order=3)
        datas_2 = butter_bandpass_filter_zi(datas[:,2], 1, 45, 100, order=3)
        datas = np.vstack([datas_0, datas_1, datas_2])
        signal_list.append(datas) 
signal_values = np.array(signal_list)
bp_signal= np.transpose(signal_values, [0, 2, 1])


#Normalized trace-by-trace
max_values_per_event = np.max(bp_signal, axis=1)
# Normalize each component of each event by dividing by its maximum value
normalized_phase_data = bp_signal / max_values_per_event[:, np.newaxis, :]
print('-----------signal format convert finish-------------------')
print(bp_signal.shape)

## Load the pre-trained model and perform the inference
learning_rate = 0.001  # Specify your learning rate

Phase_model = load_model('../Model/HANet_10w_random_1213_256_50.h5') 
phase_output = Phase_model.predict(normalized_phase_data)
P_phase_output = phase_output[:, :, 0]
S_phase_output = phase_output[:, :, 1]

## Calculating and saving the predicted error of the proposed model
Proposed_P_error_indx = evaluate_picking(P_phase_output, P_phase_label)
Proposed_S_error_indx = evaluate_picking(S_phase_output, S_phase_label)

## Load the results of EQCCT and the proposed method
# Load the CSV file (Please change the path to the output of EQCCT)
df_p = pd.read_csv('./test_pwave_outputs/X_test_results.csv')
df_s = pd.read_csv('./test_swave_outputs/X_test_results.csv')

# Filter out rows where 'P_error' is NaN
filtered_df_p = df_p[df_p['P_error'].notna()]
filtered_df_s = df_s[df_s['s_error'].notna()]

# Convert the 'P_error' column to a NumPy array
EQCCT_P_error_indx = filtered_df_p['P_error'].to_numpy()
EQCCT_S_error_indx = filtered_df_s['s_error'].to_numpy()

print(f'EQCCT P-wave picking:{EQCCT_P_error_indx.shape[0]}\nS-wave picking:{EQCCT_S_error_indx.shape[0]}')

## Load the picking results of the retrained EQCCT (Train with the same datasets as the proposed model)
df_p_r = pd.read_csv('./test_pwave_retrain_outputs/X_test_results.csv')
df_s_r = pd.read_csv('./test_swave_retrain_outputs/X_test_results.csv')

# Filter out rows where 'P_error' is NaN
filtered_df_p_r = df_p_r[df_p_r['P_error'].notna()]
filtered_df_s_r = df_s_r[df_s_r['s_error'].notna()]

# Convert the 'P_error' column to a NumPy array
EQCCT_P_error_indx_r = filtered_df_p_r['P_error'].to_numpy()
EQCCT_S_error_indx_r = filtered_df_s_r['s_error'].to_numpy()

print(f'Retrained EQCCT P-wave picking:{EQCCT_P_error_indx_r.shape[0]}\nS-wave picking:{EQCCT_S_error_indx_r.shape[0]}')

## evalute the performance of each method
xx = 3
P_error_array_eqcct, mae_P_eqcct, std_P_eqcct, precision_P_eqcct = cal_mae_std(EQCCT_P_error_indx, xx)
S_error_array_eqcct, mae_S_eqcct, std_S_eqcct, precision_S_eqcct = cal_mae_std(EQCCT_S_error_indx, xx)
P_error_proposed, mae_P_proposed, std_P_proposed, precision_P_proposed = cal_mae_std(Proposed_P_error_indx, xx)
S_error_proposed, mae_S_proposed, std_S_proposed, precision_S_proposed = cal_mae_std(Proposed_S_error_indx, xx)
P_error_array_eqcct_r, mae_P_eqcct_r, std_P_eqcct_r, precision_P_eqcct_r = cal_mae_std(EQCCT_P_error_indx_r, xx)
S_error_array_eqcct_r, mae_S_eqcct_r, std_S_eqcct_r, precision_S_eqcct_r = cal_mae_std(EQCCT_S_error_indx_r, xx)

print(f'P-wave MAE and sigma of proposed: {mae_P_proposed:.4f} s\t{std_P_proposed:.4f} s\nP-wave MAE and sigma of EQCCT   : {mae_P_eqcct:.4f} s\t{std_P_eqcct:.4f} s\nP-wave MAE and sigma of retrained EQCCT   : {mae_P_eqcct_r:.4f} s\t{std_P_eqcct_r:.4f} s')
print(f'S-wave MAE and sigma of proposed: {mae_S_proposed:.4f} s\t{std_S_proposed:.4f} s\nS-wave MAE and sigma of EQCCT   : {mae_S_eqcct:.4f} s\t{std_S_eqcct:.4f} s\nS-wave MAE and sigma of retrained EQCCT   : {mae_S_eqcct_r:.4f} s\t{std_S_eqcct_r:.4f} s')
print(f'P-wave precision of proposed: {precision_P_proposed:.2f}%\tS-wave precision of proposed: {precision_S_proposed:.2f}%')
print(f'P-wave precision of EQCCT: {precision_P_eqcct:.2f}%\tS-wave precision of EQCCT: {precision_S_eqcct:.2f}%')
print(f'P-wave precision of retrained EQCCT: {precision_P_eqcct_r:.2f}%\tS-wave precision of retrained EQCCT: {precision_S_eqcct_r:.2f}%')


## Results visualization
plt.figure(figsize=(13, 6))  # Set figure size
ax = plt.subplot(231)
plt.hist(P_error_array_eqcct, bins=30, color='palegoldenrod', edgecolor='black', alpha=0.7)  # Adjust bins, color, edgecolor, and transparency
plt.xlabel('Picking error (P)', fontsize=12, fontweight='bold')  # Add x-axis label with custom font size
plt.ylabel('Count', fontsize=12, fontweight='bold')  # Add y-axis label with custom font size
plt.xticks(fontsize=10, fontweight='bold')  # Set font size for x-axis ticks
plt.yticks(fontsize=10, fontweight='bold')  # Set font size for y-axis ticks
plt.text(0.2, 0.8, f'MAE={mae_P_eqcct:.2f} s\n$\sigma$={std_P_eqcct:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines with custom style and transparency
plt.legend(['EQCCT (P)'], prop={'weight': 'bold'})

ax = plt.subplot(232)
plt.hist(P_error_proposed, bins=30, color='skyblue', edgecolor='black', alpha=0.7)  # Adjust bins, color, edgecolor, and transparency
plt.xlabel('Picking error (P)', fontsize=12, fontweight='bold')  # Add x-axis label with custom font size
plt.ylabel('Count', fontsize=12, fontweight='bold')  # Add y-axis label with custom font size
plt.xticks(fontsize=10, fontweight='bold')  # Set font size for x-axis ticks
plt.yticks(fontsize=10, fontweight='bold')  # Set font size for y-axis ticks
plt.text(0.2, 0.8, f'MAE={mae_P_proposed:.2f} s\n$\sigma$={std_P_proposed:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines with custom style and transparency
plt.legend(['Proposed (P)'], prop={'weight': 'bold'})

ax = plt.subplot(233)
plt.hist(P_error_array_eqcct_r, bins=30, color='tomato', edgecolor='black', alpha=0.7)  # Adjust bins, color, edgecolor, and transparency
plt.xlabel('Picking error (P)', fontsize=12, fontweight='bold')  # Add x-axis label with custom font size
plt.ylabel('Count', fontsize=12, fontweight='bold')  # Add y-axis label with custom font size
plt.xticks(fontsize=10, fontweight='bold')  # Set font size for x-axis ticks
plt.yticks(fontsize=10, fontweight='bold')  # Set font size for y-axis ticks
plt.text(0.2, 0.8, f'MAE={mae_P_eqcct_r:.2f} s\n$\sigma$={std_P_eqcct_r:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines with custom style and transparency
plt.legend(['EQCCT_R (P)'], prop={'weight': 'bold'})

ax = plt.subplot(234)
plt.hist(S_error_array_eqcct, bins=30, color='palegoldenrod', edgecolor='black', alpha=0.7)  # Adjust bins, color, edgecolor, and transparency
plt.xlabel('Picking error (s)', fontsize=12, fontweight='bold')  # Add x-axis label with custom font size
plt.ylabel('Count', fontsize=12, fontweight='bold')  # Add y-axis label with custom font size
plt.xticks(fontsize=10, fontweight='bold')  # Set font size for x-axis ticks
plt.yticks(fontsize=10, fontweight='bold')  # Set font size for y-axis ticks
plt.text(0.2, 0.8, f'MAE={mae_S_eqcct:.2f} s\n$\sigma$={std_S_eqcct:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines with custom style and transparency
plt.legend(['EQCCT (S)'], prop={'weight': 'bold'})

ax = plt.subplot(235)
plt.hist(S_error_proposed, bins=30, color='skyblue', edgecolor='black', alpha=0.7)  # Adjust bins, color, edgecolor, and transparency
plt.xlabel('Picking error (s)', fontsize=12, fontweight='bold')  # Add x-axis label with custom font size
plt.ylabel('Count', fontsize=12, fontweight='bold')  # Add y-axis label with custom font size
plt.xticks(fontsize=10, fontweight='bold')  # Set font size for x-axis ticks
plt.yticks(fontsize=10, fontweight='bold')  # Set font size for y-axis ticks
plt.text(0.2, 0.8, f'MAE={mae_S_proposed:.2f} s\n$\sigma$={std_S_proposed:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
# plt.text(0.2, 0.8, f'MAE={mae_S_proposed:.2f} s\n$\sigma$={std_S_proposed:.2f} s', fontsize=10, fontweight='bold', transform=plt.gcf().transFigure)
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines with custom style and transparency
plt.legend(['Proposed (S)'], prop={'weight': 'bold'})

ax = plt.subplot(236)
plt.hist(S_error_array_eqcct_r, bins=30, color='tomato', edgecolor='black', alpha=0.7)  # Adjust bins, color, edgecolor, and transparency
plt.xlabel('Picking error (s)', fontsize=12, fontweight='bold')  # Add x-axis label with custom font size
plt.ylabel('Count', fontsize=12, fontweight='bold')  # Add y-axis label with custom font size
plt.xticks(fontsize=10, fontweight='bold')  # Set font size for x-axis ticks
plt.yticks(fontsize=10, fontweight='bold')  # Set font size for y-axis ticks
plt.text(0.2, 0.8, f'MAE={mae_S_eqcct_r:.2f} s\n$\sigma$={std_S_eqcct_r:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines with custom style and transparency
plt.legend(['EQCCT_R (S)'], prop={'weight': 'bold'})

plt.tight_layout()

plt.savefig(f'./figs/EQCCT_vs_proposed.pdf', bbox_inches="tight", dpi=150)
plt.savefig(f'./figs/EQCCT_vs_proposed.png', bbox_inches="tight", dpi=150)
plt.show()

