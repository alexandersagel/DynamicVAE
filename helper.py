import numpy as np
import scipy.misc as scm
import torch
from torch.utils.data import TensorDataset
import vae
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import imageio
import torchvision.models as models
import torch.nn.functional as F

'''
helper.py
Author: Alexander Sagel
Contact: a.sagel@tum.de

This module provides the majority of the functions to run the experiments
'''


def create_doubles(frames_list):
    '''
    Helper function for creating MNIST frame pairs
    '''
    dframes_list = []
    for frms in frames_list:
        if len(frms.shape) < 4:
            frms = frms.reshape(frms.shape[0], 1,
                                frms.shape[1], frms.shape[2])
        dframes_list.append(frms.repeat((1, 1, 2, 1)))
    dframes = torch.cat(
            dframes_list, dim=1).view(-1, 1, frms.size(2), frms.size(3))
    return dframes[1:-1].view(-1, 2, frms.size(2), frms.size(3))


def to_var(x, requires_grad=False):
    '''Variable conversion for GPU computation (for older PyTorch versions)'''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def readvid_gr(filename,  scale=1.0):
    '''
    this function returns a numpy array containing the video frames of the
    provided avi file converted to grayscale and scaled by the indicated
    factor.

    Output array has the dimensions (video length, height, width)
    '''
    V = []
    vid = imageio.get_reader(filename,  'ffmpeg')

    for image in vid.iter_data():
        arr = np.float32(scm.imresize(np.asarray(image), scale))/255
        grarr = 0.299*arr[:, :, 0] + 0.587*arr[:, :, 1] + 0.114*arr[:, :, 2]
        V.append(np.expand_dims(grarr, 0))
    fr = vid.get_meta_data()['fps']
    vid.close()
    return [np.concatenate(V), fr]


def readvid(filename,  scale=1.0):
    '''
    this function returns a numpy array containing the video frames of the
    provided avi file converted scaled by the indicated factor.

    Output array has the dimensions (video length, height, width)
    '''
    V = []
    vid = imageio.get_reader(filename,  'ffmpeg')

    for image in vid.iter_data():
        arr = np.float32(scm.imresize(np.asarray(image), scale))/255
        rgbarr = np.zeros((1, 3, arr.shape[0], arr.shape[1]))
        rgbarr[0, 0] = arr[:, :, 0].copy()
        rgbarr[0, 1] = arr[:, :, 1].copy()
        rgbarr[0, 2] = arr[:, :, 2].copy()
        V.append(rgbarr)
    fr = vid.get_meta_data()['fps']
    vid.close()
    return [np.concatenate(V), fr]


def writevid(V, filename, fps=15, scale=1.0):
    vid = imageio.get_writer(filename, 'ffmpeg', fps=fps)
    for v in V:
        if len(v.shape) > 2:
            rgbv = np.zeros((v.shape[1], v.shape[2], 3))
            rgbv[:, :, 0] = v[0]
            rgbv[:, :, 1] = v[1]
            rgbv[:, :, 2] = v[2]
            vid.append_data(scm.imresize(np.uint8(rgbv*255.0), scale))
        else:
            vid.append_data(np.uint8(v*255.0))
    vid.close()


def print_params(filename, p):
    '''
    This function prints the parameter dictionary p to the provided textfile
    path
    '''
    # FIXME: Inclue c for WGAN
    p_file = open(filename, "w")
    p_file.write('Network type: ' + str(p['network_type'])+'\n')
    p_file.write('Latent dimension: ' + str(p['latent_dim'])+'\n')
    p_file.write('Learning rate: '+str(p['learning_rate'])+'\n')
    p_file.write('# of epochs: '+str(p['n_epochs'])+'\n')
    p_file.write('# of Monte Carlo samples: '+str(p['n_mc'])+'\n')
    p_file.write('# of convolutional layers: '+str(p['n_clayers'])+'\n')
    p_file.write('# size of first filter kernel: '+str(p['kernel1_size'])+'\n')
    p_file.write('sigma_w^2: '+str(p['sigma_squared'])+'\n')
    p_file.write('lambda: '+str(p['lambda'])+'\n')
    p_file.close()


def save_gt(data, file_prefix):
    '''
    Saves the provided Training Sequence
    '''
    
    data = data.numpy()
    Y = data[:, :np.uint8(data.shape[1]/2)]+1
    writevid(Y.squeeze()/2, file_prefix + '.avi', 15)


def run_experiment(arg1, arg2, arg3, arg4=None):
    '''
    Container Function for running an experiment
    '''
    
    ret = False
    if arg4 is None:
        if arg3['network_type'] == 'VAE':
            ret = run_experiment_vae(arg1, arg2, arg3)
        elif arg3['network_type'] == 'DVAE':
            ret = run_experiment_dvae(arg1, arg2, arg3)
        elif arg3['network_type'] == 'Linear':
            ret = run_experiment_linear(arg1, arg2, arg3)
    else:
        if arg4['incomplete'] == 'incomplete':
            ret = run_experiment_incomplete(arg1, arg2, arg3, arg4)
        else:
            print('Network type not recognized!')
    return ret


def run_experiment_linear(data, file_prefix, p):
    '''
    Training and Synthesis via the original LDS model by Doretto et al.
    '''
    
    data = data.numpy()
    y_prev = data[:, :np.uint8(data.shape[1]/2)].reshape(data.shape[0], -1)
    y_next = data[:, np.uint8(data.shape[1]/2):].reshape(data.shape[0], -1)

    y_mu = np.expand_dims(np.mean(y_prev[1:], axis=0), 0)
    Uc, Sc, VcT = np.linalg.svd(y_prev[1:]-y_mu, full_matrices=False)

    C = VcT[:p['latent_dim'], :].T

    h_prev = np.dot(y_prev-y_mu, C)
    h_next = np.dot(y_next-y_mu, C)

    A = np.dot(np.dot(h_next.T, h_prev), np.linalg.inv(np.dot(h_prev.T,
               h_prev)))
    Ua, Sa, VaT = np.linalg.svd(A)
    Sa = np.where(Sa > 1.0, 1.0, Sa)
    A = np.dot(Ua*Sa, VaT)
    Ub, Sb, _ = np.linalg.svd(h_next.T-np.dot(A, h_prev.T))
    B = np.dot(Ub, np.diag(Sb))/np.sqrt(h_next.shape[0])
    Z = np.zeros((p['N_synth'], p['latent_dim']))
    Z[0] = h_prev[0]
    for i in range(p['N_synth']-1):
        Z[i+1] = np.dot(A, Z[i]) + np.dot(B, np.random.randn(p['latent_dim']))

    Y = ((np.dot(Z, C.T)+y_mu).reshape(-1, np.uint8(data.shape[1]/2),
                                       data.shape[2], data.shape[3])+1)/2
    Y = np.where(Y < 0, 0, Y)
    Y = np.where(Y > 1, 1, Y)
    torchvision.utils.save_image(torch.from_numpy(Y[:p['n_row']*p['n_col']]),
                                 file_prefix + '.png',
                                 nrow=p['n_row'])
    if p['create_video']:
        writevid(Y.squeeze(), file_prefix + '.avi', p['fps'])
    return True


def run_experiment_vae(data, file_prefix, p):
    '''
    Synthesis via training the VAE and the VAR separately
    '''
    
    y = torch.zeros(data.size(0)).float()
    dataset = TensorDataset(data[1:, [0], :, :], y[1:])
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=True)
    net = vae.VAE(latent_dim=p['latent_dim'],
                  n_clayers=p['n_clayers'], kernel1_size=p['kernel1_size'])
    optimizer = torch.optim.Adam(net.parameters(), lr=p['learning_rate'])
    test_noise = torch.randn(p['N_synth'], p['latent_dim'])
    mse_criterion = MSELoss(reduction='sum')
    for epoch in range(p['n_epochs']):
        cst = 0
        kld = 0
        for i, (images, _) in enumerate(data_loader):
            images = to_var(images)
            out, mu, log_var = net(images, N=p['n_mc'])
            reconst_loss = mse_criterion(out, images.repeat(
                    p['n_mc'], 1, 1, 1))/p['n_mc']
            try:
                kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var)
                                          - log_var-1))
            except RuntimeError:
                return False

            total_loss = reconst_loss/(p['sigma_squared']) + kl_divergence
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            cst = total_loss + cst
            kld = kl_divergence + kld
        if epoch % p['save_interval'] == 0:
            net.eval()
            print("Epoch:", epoch+1, "- Averaged Cost:", cst.item()/(i+1),
                  'kld:', kld.item()/(i+1))
            h_prev = np.zeros((data.size(0)-1, p['latent_dim']))
            h_next = np.zeros((data.size(0)-1, p['latent_dim']))
            for i in range(data.size(0)-1):
                mu_synth_prev, _ = torch.chunk(net.encoder(to_var(
                        data[[i]][:, [0], :, :])), 2, dim=1)
                h_prev[i] = mu_synth_prev.data.cpu().numpy()
                mu_synth_next, _ = torch.chunk(net.encoder(to_var(
                        data[[i]][:, [1], :, :])), 2, dim=1)
                h_next[i] = mu_synth_next.data.cpu().numpy()
            A = np.dot(np.dot(h_next.T, h_prev), np.linalg.inv(np.dot(h_prev.T,
                       h_prev)))
            Ua, Sa, VaT = np.linalg.svd(A)
            Sa = np.where(Sa > 1.0, 1.0, Sa)
            A = np.dot(Ua*Sa, VaT)
            Ub, Sb, _ = np.linalg.svd(np.eye(p['latent_dim'])-np.dot(A, A.T))
            B = np.dot(Ub, np.diag(np.sqrt(Sb)))
            Y = (net.synthesize(torch.from_numpy(A).float(),
                 torch.from_numpy(B).float(),
                 additive_noise=test_noise,
                 img_init=data[[0]][:, [0], :, :])+1)/2
            torchvision.utils.save_image(Y[:p['n_row']*p['n_col']].data.cpu(),
                                         file_prefix + '_%03d' % epoch + '.png',
                                         nrow=p['n_row'])
            if p['create_video']:
                writevid(Y.data.cpu().numpy().squeeze(),
                         file_prefix+'_%03d' % epoch + '.avi', p['fps'])
            net.train()
    return True


def run_experiment_dvae(sdata, file_prefix, p):
    '''
    Training and Synthesis via the DVAE
    '''
    y = torch.zeros(sdata.size(0)).float()
    dataset = TensorDataset(sdata[1:], y[1:])
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=True)

    otpsz = p['kernel1_size']*(2**(p['n_clayers']-1))
    ds = max(sdata.size(-1)/otpsz, 1)
    dvae = vae.DVAE(latent_dim=p['latent_dim'],
                    n_clayers=p['n_clayers'], kernel1_size=p['kernel1_size'],
                    n_channels=int(sdata.size(1)/2), ds=ds)

    optimizer = torch.optim.Adam(dvae.parameters(), lr=p['learning_rate'])
    mse_criterion = MSELoss(reduction='sum')
    IDTY = to_var(torch.eye(p['latent_dim']), requires_grad=False)
    with torch.no_grad():
        test_noise = torch.randn(p['N_synth'], p['latent_dim'])
    for epoch in range(p['n_epochs']):
        cst = 0
        kld = 0
        for i, (images, _) in enumerate(data_loader):
            images = to_var(images)
            out, mu, log_var = dvae(images, p['n_mc'])
            reconst_loss = mse_criterion(out, images.repeat(
                    p['n_mc'], 1, 1, 1)) / p['n_mc']
            try:
                kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var)
                                          - log_var-1))
                reg = torch.sum(torch.pow((torch.mm(dvae.A, torch.t(dvae.A))
                                           + torch.mm(dvae.B, torch.t(dvae.B)))
                                          - IDTY, 2))
            except RuntimeError:
                return False
        # Backprop + Optimize
            half_loss = reconst_loss/p['sigma_squared'] + kl_divergence
            total_loss = half_loss+p['lambda']*reg
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            cst += total_loss
            kld += kl_divergence
        if np.isnan(cst.item()) or np.isnan(cst.item()):
            return False
        if epoch % p['save_interval'] == 0:
            dvae.eval()
            print("Epoch:", epoch+1, "- Averaged Cost:", cst.item()/(i+1),
                  'kld:', kld.item()/(i+1),
                  'Regularizer Cost', reg.item())
            Y = (dvae.synthesize(img_init=sdata[[0]],
                                 additive_noise=test_noise)+1)/2
            torchvision.utils.save_image(Y[:p['n_row']*p['n_col']].data.cpu(),
                                         file_prefix+'_%03d' % epoch+'.png',
                                         p['n_row'])
            torchvision.utils.save_image(dvae.decoder(to_var(
                                  torch.zeros(1, p['latent_dim']))).data.cpu(),
                                  file_prefix + '_fp.png', 1)
            if p['create_video']:
                writevid(Y.data.cpu().numpy().squeeze(),
                         file_prefix+'_%03d' % epoch + '.avi', p['fps'])
            dvae.train()
    dvae = None
    return True


def run_experiment_incomplete(sdata, smask, file_prefix, p):
    '''
    Performing training and synthesis of one sequence
    '''

    # preparing data
    y = torch.zeros(sdata.size(0)).float()
    dataset = TensorDataset(torch.cat((sdata, smask), dim=1)[1:], y[1:])
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=True)
    n_channels = int(sdata.size(1)/2)
    dvae = vae.DVAE(latent_dim=p['latent_dim'],
                    n_clayers=p['n_clayers'], kernel1_size=p['kernel1_size'],
                    n_channels=n_channels, incomplete=True)

    optimizer = torch.optim.Adam(dvae.parameters(), lr=p['learning_rate'])
    mse_criterion = MSELoss(reduction='sum')
    IDTY = to_var(torch.eye(p['latent_dim']), requires_grad=False)
    with torch.no_grad():
        test_noise = torch.randn(p['N_synth'], p['latent_dim'])
    for epoch in range(p['n_epochs']):
        cst = 0
        kld = 0
        for i, (images, _) in enumerate(data_loader):
            images = to_var(images)
            out, mu, log_var = dvae(images, p['n_mc'])
            current_mask = torch.cat((images[:, [2*n_channels], :, :].repeat(1,
                                      n_channels, 1, 1),
                                      images[:, [2*n_channels+1], :, :].repeat(
                                              1, n_channels, 1, 1)), dim=1)
            reconst_loss = mse_criterion(out*current_mask.repeat(p['n_mc'], 1,
                                                                 1, 1),
                                         (images[:, :2*n_channels, :, :]
                                         * current_mask).repeat(
                                                 p['n_mc'],
                                                 1, 1, 1)) / p['n_mc']
            try:
                kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var)
                                          - log_var-1))
                reg = torch.sum(torch.pow((torch.mm(dvae.A, torch.t(dvae.A))
                                           + torch.mm(dvae.B, torch.t(dvae.B)))
                                          - IDTY, 2))
            except RuntimeError:
                return False
        # Backprop + Optimize
            half_loss = reconst_loss/p['sigma_squared'] + kl_divergence
            total_loss = half_loss+p['lambda']*reg
        #    print(reg.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            cst += total_loss
            kld += kl_divergence
        if np.isnan(cst.item()) or np.isnan(cst.item()):
            return False
        if epoch % p['save_interval'] == 0:
            dvae.eval()
            print("Epoch:", epoch+1, "- Averaged Cost:", cst.item()/(i+1),
                  'kld:', kld.item()/(i+1),
                  'Regularizer Cost', reg.item())
            Y = (dvae.synthesize(additive_noise=test_noise)+1)/2
            torchvision.utils.save_image(Y[:p['n_row']*p['n_col']].data.cpu(),
                                         file_prefix+'_%03d' % epoch+'.png',
                                         p['n_row'])
            torchvision.utils.save_image(dvae.decoder(to_var(
                                  torch.zeros(1, p['latent_dim']))).data.cpu(),
                                  file_prefix + '_fp.png', 1)
            if p['create_video']:
                writevid(Y.data.cpu().numpy().squeeze(),
                         file_prefix+'_%03d' % epoch + '.avi', p['fps'])
            dvae.train()
    dvae = None
    return True


def normal_init(m, mean, std):
    '''
    Weight initialization
    '''
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
