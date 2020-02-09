#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Alexander Sagel
Contact: a.sagel@tum.de

Code for reproducing the Dynamic Texture experiments with DVAE
"""

import helper
import torch
import numpy as np
import glob
import os
from os import path
import urllib.request
import zipfile
import datetime

if not(path.exists('./data/stgconv_data')):
    print('Downloading STGConvNet_code.zip...')
    url = 'http://www.stat.ucla.edu/~jxie/STGConvNet/STGConvNet_file/code/STGConvNet_code.zip'
    urllib.request.urlretrieve(url, './data/STGConvNet_code.zip')
    print('Unzipping...')
    zf = zipfile.ZipFile('./data/STGConvNet_code.zip')
    os.mkdir('./data/stgconv_data')
    for file in zf.namelist():
        if file.startswith('STGConvNet_code/trainingVideo/data_synthesis/'):
            zf.extract(file, './data/')
            if file.endswith('.avi'):
                dirsplit = file.split('/')
                os.rename('data/'+file,
                          'data/stgconv_data/'+dirsplit[-1])
    print('Done!')

# Setting experimental parameters
experiment_parameters = {}
experiment_parameters = {}
experiment_parameters['latent_dim'] = 10
experiment_parameters['n_mc'] = 128
experiment_parameters['sigma_squared'] = 32
experiment_parameters['lambda'] = 100
experiment_parameters['n_epochs'] = 101
experiment_parameters['learning_rate'] = 0.0001
experiment_parameters['create_video'] = True
experiment_parameters['N_synth'] = 600
experiment_parameters['n_row'] = 9
experiment_parameters['n_col'] = 6
experiment_parameters['network_type'] = 'DVAE'
experiment_parameters['save_interval'] = 5
experiment_parameters['n_clayers'] = 5
experiment_parameters['kernel1_size'] = 8

# Setting up data folder
resultdir = './results/stgconv_data/'
if not(path.exists(resultdir)):
    os.mkdir(resultdir)
datestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(resultdir+datestring)


# Running the experiments
files = glob.glob('./data/stgconv_data/*.avi')
files.sort()
for f_i, file in enumerate(files):
    V, fr = helper.readvid(file, scale=(128, 128))
    os.mkdir(resultdir+datestring+'/v'+str(f_i))
    dpref = resultdir+datestring+'/v'+str(f_i)
    fpref = dpref+'/vid'+str(f_i)
    print('Processing now:', fpref)
    print()
    V_ = np.zeros((V.shape[0]-1, 6, V.shape[2], V.shape[3]))
    V_[:, 0, :, :] = V[:-1, 0]
    V_[:, 1, :, :] = V[:-1, 1]
    V_[:, 2, :, :] = V[:-1, 2]
    V_[:, 3, :, :] = V[1:, 0]
    V_[:, 4, :, :] = V[1:, 1]
    V_[:, 5, :, :] = V[1:, 2]

    experiment_parameters['fps'] = fr
    experiment_parameters['n_epochs'] = 62500 // V.shape[0]
    experiment_parameters['save_interval'
                          ] = experiment_parameters['n_epochs'] // 20
    helper.print_params(dpref + '/PARAMETERS.txt',
                        experiment_parameters)
    npdata = V_*2-1
    traindata = torch.from_numpy(npdata).float()
    ret = helper.run_experiment(traindata, fpref, experiment_parameters)
    if not ret:
        print('Could not synthesize', fpref, 'due to runtime error')
