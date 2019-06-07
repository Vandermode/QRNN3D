"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center, Visualize3D, minmax_normalize


def create_mat_dataset(datadir, fnames, newdir, matkey, func=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        mat = load(filepath)
        
        data = func(mat[matkey][...])
        # Visualize3D(data)
        # import ipdb; ipdb.set_trace()

        # if not exists(join(newdir, fn)):
        savemat(join(newdir, fn), {'data':data.transpose((2,1,0))})


def create_icvl_sr():
    basedir = '/media/kaixuan/DATA/Papers/Code/Matlab/ITSReg/code of ITSReg MSI denoising/data'
    datadir = join(basedir, 'icvl_test')
    newdir = join(basedir, 'icvl_256_sr')
    fnames = os.listdir(datadir)
    
    def func(data):
        data = np.rot90(data, k=-1, axes=(1,2))
        
        data = crop_center(data, 256, 256)
        
        data = minmax_normalize(data)
        return data
    
    create_mat_dataset(datadir, fnames, newdir, 'rad', func=func)


if __name__ == '__main__':
    # create_icvl_sr()
    pass
