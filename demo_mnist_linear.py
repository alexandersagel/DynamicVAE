#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Alexander Sagel
Contact: a.sagel@tum.de

Code for reproducing the MNIST experiment with linear model by Doretto et al.
"""
import numpy as np
import torchvision
from torchvision import datasets, transforms
import os
import datetime
import helper
import torch
from os import path



# Setting experimental parameters
experiment_parameters = {}
experiment_parameters['latent_dim'] = 10
experiment_parameters['create_video'] = True
experiment_parameters['N_synth'] = 70
experiment_parameters['n_row'] = 10
experiment_parameters['n_col'] = 7
experiment_parameters['fps'] = 5
experiment_parameters['network_type'] = 'Linear'

# Retrieving the data
mnist = datasets.MNIST('./data', download=True)
rawdata = mnist.train_data
trainlabels = mnist.train_labels

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])


# Setting up data folder
resultdir = './results/mnist_linear/'
if not(path.exists(resultdir)):
    os.mkdir(resultdir)
datestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

dpref = resultdir+datestring
os.mkdir(dpref)


# Sequences to synthesize
dg_l = [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]


# Running the experiments
for dg_seq in dg_l:
    frames_list = []
    seqstr = ''
    for framesnum in dg_seq:
        idx = (trainlabels == framesnum).nonzero()
        frames_list.append(rawdata[idx].float()[:2000]/255)
        seqstr += str(framesnum)
    sdata = helper.create_doubles(frames_list)
    traindata = torch.zeros(sdata.shape[0], 2, 32, 32)
    for i in range(sdata.shape[0]):
        traindata[i][[0]] = transform((sdata[i][[0]]).float())
        traindata[i][[1]] = transform((sdata[i][[1]]).float())
    os.mkdir(dpref+'/mnist_'+seqstr)
    file_prefix = dpref+'/mnist_'+seqstr+'/mnist_'+seqstr

    print('Processing now:', file_prefix + '...')
    print()
    ret = helper.run_experiment(traindata, file_prefix, experiment_parameters)

    if not ret:
        # try again once
        ret = helper.run_experiment(traindata, file_prefix,
                                    experiment_parameters)
    if not ret:
        print('Could not synthesize', file_prefix, 'due to runtime error')
