import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options
import models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


prefix = 'test'

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    print(opt)

    cuda = not opt.no_cuda
    opt.no_log = True

    """Setup Engine"""
    engine = Engine(opt)

    datadir = '' # your input data dir
    basefolder = '/media/kaixuan/DATA/Papers/Code/Matlab/ECCV2018/ECCVData'
    # datadir = os.path.join(basefolder, 'icvl_512_50')
    # datadir = os.path.join(basefolder, 'icvl_512_blind')
    # datadir = os.path.join(basefolder, 'icvl_512_noniid')
    datadir = os.path.join(basefolder, 'icvl_512_mixture')
    
    mat_dataset = MatDataFromFolder(datadir, size=None)

    # mat_dataset.filenames = [
    #         os.path.join(datadir, 'Lehavim_0910-1627.mat') 
    #     ]
    
    mat_transform = Compose([
        LoadMatHSI(input_key='input', gt_key='gt', transform=lambda x:x[:,:,:][None]), # for validation
        # LoadMatKey(key='hsi'), # for testing
        # lambda x: x[None]
    ])

    mat_dataset = TransformDataset(mat_dataset, mat_transform)
    mat_loader = DataLoader(
                    mat_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=cuda
                )
    
    resdir = None # your result dir
    
    # res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
    # print(res_arr.mean(axis=0))
    engine.validate(mat_loader, '')