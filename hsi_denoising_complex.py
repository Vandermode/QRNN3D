import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options, make_dataset

sigmas = [10, 30, 50, 70]

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    add_noniid_noise = Compose([
        AddNoiseNoniid(sigmas),
        SequentialSelect(
            transforms=[
                lambda x: x,
                AddNoiseImpulse(),
                AddNoiseStripe(),
                AddNoiseDeadline()
            ]
        )
    ])

    common_transform = Compose([
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = HSI2Tensor()

    train_transform = Compose([
        add_noniid_noise,
        HSI2Tensor()
    ])

    print('==> Preparing data..')

    icvl_64_31_TL = make_dataset(
        opt, train_transform,
        target_transform, common_transform, 64)

    """Test-Dev"""
    basefolder = '/data/weikaixuan/hsi/data/'
    mat_names = ['icvl_512_noniid', 'icvl_512_mixture']

    mat_datasets = [MatDataFromFolder(os.path.join(
        basefolder, name), size=5) for name in mat_names]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[:, ...][None]),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt'),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]        

    base_lr = 1e-3
    epoch_per_save = 10
    adjust_learning_rate(engine.optimizer, opt.lr)
    # from epoch 50 to 100
    while engine.epoch < 100:
        np.random.seed()

        if engine.epoch == 85:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)
        
        if engine.epoch == 95:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
        
        engine.train(icvl_64_31_TL)

        engine.validate(mat_loaders[0], 'icvl-validate-noniid')
        engine.validate(mat_loaders[1], 'icvl-validate-mixture')

        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()
