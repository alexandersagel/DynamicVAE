#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Alexander Sagel
Contact: a.sagel@tum.de
Code for reproducing the incomplete windmill experiment with DVAE
"""

import helper
import torch
import numpy as np
import os
from os import path
import urllib.request
import datetime

# Retrieving the data.
if not(path.exists('./data/incomplete')):
    os.mkdir('./data/incomplete')
    print('Downloading windmill_incomplete.avi...')
    url = 'https://www.dropbox.com/s/5j5ul3kbakuljbp/windmill_incomplete.avi?dl=1'
    urllib.request.urlretrieve(url,
                               './data/incomplete/windmill_incomplete.avi')
    print('Downloading windmill_mask.avi...')
    url = 'https://www.dropbox.com/s/200woutgt34hywy/windmill_mask.avi?dl=1'
    urllib.request.urlretrieve(url, './data/incomplete/windmill_mask.avi')

# Setting experimental parameters
experiment_parameters = {}
experiment_parameters = {}
experiment_parameters['latent_dim'] = 10
experiment_parameters['n_mc'] = 128
experiment_parameters['sigma_squared'] = 64
experiment_parameters['lambda'] = 100
experiment_parameters['n_epochs'] = 201
experiment_parameters['learning_rate'] = 0.0001
experiment_parameters['create_video'] = True
experiment_parameters['N_synth'] = 2000
experiment_parameters['n_row'] = 9
experiment_parameters['n_col'] = 6
experiment_parameters['network_type'] = 'incomplete'
experiment_parameters['save_interval'] = 5
experiment_parameters['fps'] = 25
experiment_parameters['n_clayers'] = 5
experiment_parameters['kernel1_size'] = 4

# Setting up data folder
resultdir = './results/incomplete/'
if not(path.exists(resultdir)):
    os.mkdir(resultdir)
datestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(resultdir+datestring)

# Saving parameter file
helper.print_params('./results/incomplete/'+datestring+'/PARAMETERS.txt',
                    experiment_parameters)

# Running the experiments
V, _ = helper.readvid('./data/incomplete/windmill_incomplete.avi')
M, _ = helper.readvid('./data/incomplete/windmill_mask.avi')
dpref = resultdir+datestring

fpref = dpref+'/'+'incomplete'
print('Processing now:', fpref)
print()
vid_pairs = []
V_ = np.zeros((V.shape[0]-1, 6, V.shape[2], V.shape[3]))
V_[:, 0, :, :] = V[:-1, 0]
V_[:, 1, :, :] = V[:-1, 1]
V_[:, 2, :, :] = V[:-1, 2]
V_[:, 3, :, :] = V[1:, 0]
V_[:, 4, :, :] = V[1:, 1]
V_[:, 5, :, :] = V[1:, 2]
M_ = np.zeros((M.shape[0]-1, 2, M.shape[2], M.shape[3]))
M_[:, 0, :, :] = M[:-1, 0]
M_[:, 1, :, :] = M[1:, 0]


npdata = V_*2-1
traindata = torch.from_numpy(npdata).float()
trainmask = torch.from_numpy(M_).float()
ret = helper.run_experiment_incomplete(traindata, trainmask,
                                       fpref, experiment_parameters)
if not ret:
    ret = helper.run_experiment_incomplete(traindata, trainmask, fpref,
                                           experiment_parameters)
    if not ret:
        print('Could not synthesize', fpref, 'due to runtime error')
