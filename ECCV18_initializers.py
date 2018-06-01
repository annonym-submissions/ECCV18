import numpy as np
from ECCV18_specify_experiment_params import *
import os
###############################
data_path = os.path.abspath('Data')
PGF_path = os.path.abspath('Results/PGF')  # That's where you store the graphs.
###########################
if Database == 'MNIST':
    data = np.load(data_path + '/mnist.npz')
    F = data['imgs_train'].reshape(784, -1).astype('float')
    Q = data['imgs_test'].reshape(784, -1).astype('float')
    F_train = F
    Y_NNs = np.load(data_path + 'mnist_Y_NNs.npz')['Y_NNs']
    trainset_mainset_seperated = False
    del data

elif Database == 'GIST1M':
    import h5py
    arrays = {}
    f = h5py.File(data_path + '/GIST1M.mat')
    for ind1, ind2 in f.items():
        arrays[ind1] = np.array(ind2)
    F = arrays['F'].T
    F_train = arrays['F_train'].T
    Q = arrays['Q'].T
    Y_NNs = arrays['Y_NNs'].T
    trainset_mainset_seperated = True
    del arrays

elif Database == 'SIFT1M':
    import h5py
    arrays = {}
    f = h5py.File(data_path + '/SIFT1M.mat')
    for ind1, ind2 in f.items():
        arrays[ind1] = np.array(ind2)
    F = arrays['F'].T
    F_train = arrays['F_train'].T
    Q = arrays['Q'].T
    Y_NNs = arrays['Y_NNs'].T
    trainset_mainset_seperated = True
    del arrays
    Q = Q[:,0::10]
    Y_NNs = Y_NNs[:,0::10]
##########################
# Pre-processing:
# This does not change the data manifold, so it is universally valid.
F -= np.mean(F)
F_train -= np.mean(F_train)
Q -= np.mean(Q)
######
F /= np.std(F)
F_train /= np.std(F_train)
Q /= np.std(Q)
# setting some intermediate params:
(n,N_f) = np.shape(F)
N_f_train = np.shape(F_train)[1]
N_q = np.shape(Q)[1]

