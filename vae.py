#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae.py
Author: Alexander Sagel
Contact: a.sagel@tum.de

Dynamic VAE Netyork implementation. Based on the vanilla VAE
implementation by Yunjey Choi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import helper

class StaticEncoder(nn.Module):
    '''Convolutional Encoder Class for the Static VAE'''
    def __init__(self, d=128, n_clayers=5, latent_dim=10, n_channels=1,
                 kernel1_size=4):
        super(StaticEncoder, self).__init__()
        self.latent_dim=latent_dim
        self.conv_first = nn.Conv2d(n_channels, d, 4, 2, 1)
        chan_in = d
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(n_clayers-2):
            chan_out = chan_in*2
            self.conv_list.append(nn.Conv2d(chan_in, chan_out, 4, 2, 1))
            self.bn_list.append(nn.BatchNorm2d(chan_out))
            chan_in = chan_out
        self.conv_last = nn.Conv2d(chan_in, 2*latent_dim, kernel1_size, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            helper.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv_first(x), 0.2)
        for i in range(len(self.conv_list)):
            x = F.leaky_relu(self.bn_list[i](self.conv_list[i](x)), 0.2)
            #x = F.leaky_relu((self.conv_list[i](x)), 0.2)
        x = self.conv_last(x)
        return x.view(-1, 2*self.latent_dim)


class Encoder(StaticEncoder):
    '''Convolutional Encoder Class for the Dynamic VAE'''
    def __init__(self, d=128, n_clayers=5, latent_dim=10, n_channels=1,
                 kernel1_size=4, ds=1):
        super(Encoder, self).__init__(d=d, n_clayers=n_clayers,
                                      latent_dim=latent_dim,
                                      n_channels=n_channels)
        self.conv_first = nn.Conv2d(2*n_channels, d, 4, 2, 1)
        chan_in = d*2**(n_clayers-2)
        self.conv_last = nn.Conv2d(chan_in, 4*latent_dim, kernel1_size, 1, 0)
        self.ds = ds

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            helper.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        if self.ds > 1:
            x = F.interpolate(x, scale_factor=1.0/self.ds, mode='bilinear')
        x = F.leaky_relu(self.conv_first(x), 0.2)
        for i in range(len(self.conv_list)):
            x = F.leaky_relu(self.bn_list[i](self.conv_list[i](x)), 0.2)
          #  x = F.leaky_relu((self.conv_list[i](x)), 0.2)
        x = self.conv_last(x)
        return x.view(-1, 4*self.latent_dim)


class Decoder(nn.Module):
    '''Convolutional Decoder Class for the Dynamic VAE'''
    def __init__(self, latent_dim=10, d=128, n_clayers=5, n_channels=1,
                 kernel1_size=4, ds=1):
        super(Decoder, self).__init__()
        self.ds = ds
        self.dense = nn.Linear(latent_dim, 100)
        self.deconv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        chan_in = int(d*2**(n_clayers-2))
        self.deconv_first = nn.ConvTranspose2d(100, chan_in, kernel1_size, 1,
                                               0)
        for i in range(n_clayers-1):
            if i < n_clayers-2:
                chan_out = int(chan_in/2)
            else:
                chan_out = n_channels
            self.bn_list.append(nn.BatchNorm2d(chan_in))
            self.deconv_list.append(
                    nn.ConvTranspose2d(chan_in, chan_out, 4, 2, 1))
            chan_in = chan_out

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            helper.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x_):
        # x = F.relu(self.deconv1(input))
        x = self.deconv_first(self.dense(x_).view(-1, 100, 1, 1))
        for i in range(len(self.deconv_list)):
            # x = self.deconv_list[i](F.relu(self.bn_list[i](x)))
            x = self.deconv_list[i](F.relu((x)))
        if self.ds > 1:
            x = F.interpolate(x, scale_factor=self.ds, mode='bilinear')
        return torch.tanh(x)


class VAE(nn.Module):
    '''Static VAE Class'''
    def __init__(self, latent_dim=10, n_channels=1, n_clayers=5,
                 kernel1_size=4):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = StaticEncoder(d=128,  n_clayers=n_clayers,
                                     n_channels=n_channels,
                                     kernel1_size=kernel1_size)
        self.decoder = Decoder(latent_dim=latent_dim, d=128,
                               n_clayers=n_clayers, n_channels=n_channels,
                               kernel1_size=kernel1_size)

        if torch.cuda.is_available():
            self.cuda()

    def reparametrize(self, b, log_a, N=8):
        """
        Reparametrization trick

        Input:
          b:     Bias of the affine reparametrization function g
          log_a: Logarithm of the diagonal of the multiplicative part
                   of the affine reparametrization function
          N:     Number of latent Monte Carlo samples

        Output:
          N latent Monte Carlo samples
        """
        eps = helper.to_var(torch.randn(N, b.size(1)))
        z = b + eps * torch.exp(log_a/2)    # 2 for convert var to std
        return z

    def forward(self, x, N):
        '''
        Forward Propagation:

        Input:
          x: Training image sample. Must have the dimensions
             (1,1,height,width)

        Output:
          Generated image sample. Dimensions (n_mc,1,height,width) where n_mc
          is the number of monte carlo samples
        '''
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)  # mean and log variance.
        z = self.reparametrize(mu, log_var, N)
        out = self.decoder(z)
        return out, mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def synthesize(self, A, B, additive_noise, img_init=None):
        '''Sequence Synthesis

        Input:
            n:        Length of sequence
            img_init: Initial sample
        '''
        n = additive_noise.size(0)
        if img_init is None:
            Z = torch.zeros(n, self.latent_dim)
            Z[0] = additive_noise[0]
            for i in range(n-1):
                Z[i+1] = torch.mv(A, Z[i]) + torch.mv(B,
                                                      additive_noise[i+1, :])
        else:
            Z = torch.zeros(n+1, self.latent_dim).float()
            mu_logvar = self.encoder(helper.to_var(img_init))
            h_prev, _ = torch.chunk(mu_logvar, 2, dim=1)
            Z[0] = h_prev.data.cpu()
            for i in range(n):
                Z[i+1] = torch.mv(A, Z[i]) + torch.mv(B, additive_noise[i, :])
        # FIXME: inference mode
        return self.decoder(helper.to_var(Z))


class DVAE(nn.Module):
    '''Dynamic VAE Class'''
    def __init__(self, latent_dim=10, n_channels=128, n_clayers=5,
                 kernel1_size=4, incomplete=False, ds = 1):
        super(DVAE, self).__init__()
        self.latent_dim = latent_dim
        A = torch.eye(latent_dim)*0.70711
        B = torch.eye(latent_dim)*0.70711
        self.B = torch.nn.Parameter(B, requires_grad=True)
        self.A = torch.nn.Parameter(A, requires_grad=True)
        if incomplete:
            self.encoder = Encoder(d=128,  n_clayers=n_clayers,
                                   n_channels=n_channels+1,
                                   kernel1_size=kernel1_size)
        else:
            self.encoder = Encoder(d=128,  n_clayers=n_clayers,
                                   n_channels=n_channels,
                                   kernel1_size=kernel1_size, ds=ds)
        self.decoder = Decoder(latent_dim=latent_dim, d=128,
                               n_clayers=n_clayers, n_channels=n_channels,
                               kernel1_size=kernel1_size, ds=ds)
        self.encoder.weight_init(mean=0.0, std=0.02)
        self.decoder.weight_init(mean=0.0, std=0.02)        
        if torch.cuda.is_available():
            self.cuda()

    def reparametrize(self, b, log_a, N=8):
        """
        Reparametrization trick

        Input:
            b:     Bias of the affine reparametrization function g
            log_a: Logarithm of the diagonal of the multiplicative part
                   of the affine reparametrization function
            N:     Number of latent Monte Carlo samples
        """
        eps = helper.to_var(torch.randn(N, b.size(1)))
        z = b + eps * torch.exp(log_a/2)    # 2 for convert var to std
        return z

    def forward(self, x, N=8):
        '''
        Forward Propagation:

        Input:
          x: Training image sample. Must have the dimensions
             (1,2,height,width)

        Output:
          Generated image sample. Dimensions (N,2,height,width)
        '''
        mu_logvar = self.encoder(x)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
        h = self.reparametrize(mu, log_var, N)
        h_prev, h_next = torch.chunk(h, 2, dim=1)
        h_next = (torch.mm(h_prev, torch.t(self.A))
                  + torch.mm(h_next, torch.t(self.B)))
        out1 = self.decoder(h_prev)
        out2 = self.decoder(h_next)
        out = torch.cat((out1, out2), dim=1)
        return out, mu, log_var

    def synthesize(self, additive_noise, img_init=None):
        '''Sequence Synthesis

        Input:
            n:        Length of sequence
            img_init: Initial sample
        '''
        A = self.A.data.cpu()
        B = self.B.data.cpu()
        n = additive_noise.size(0)
        if img_init is None:
            Z = torch.zeros(n, self.latent_dim)
            Z[0] = additive_noise[0]
            Y = [self.decoder(helper.to_var(additive_noise[[0]])).data.cpu()]
            for i in range(n-1):
                Z[i+1] = torch.mv(A, Z[i]) + torch.mv(B,
                                                      additive_noise[i+1, :])
                Y.append(self.decoder(helper.to_var(Z[[i+1]])).data.cpu())
        else:
            Z = torch.zeros(n+1, self.latent_dim)
            mu_logvar = self.encoder(helper.to_var(img_init))
            mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
            h_prev, _ = torch.chunk(mu, 2, dim=1)
            Z[0] = h_prev.data.cpu()
            Y = [self.decoder(helper.to_var(h_prev.view(1, -1))).data.cpu()]
            for i in range(n):
                Z[i+1] = torch.mv(A, Z[i]) + torch.mv(B, additive_noise[i, :])
                Y.append(self.decoder(helper.to_var(Z[[i+1]])).data.cpu())
        # FIXME: inference mode
        return torch.cat(Y, dim=0)

    def decode(self, z):
        return self.decoder(z)
