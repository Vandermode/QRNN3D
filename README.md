# QRNN3D

The implementation of TNNLS 2020 paper ["3D Quasi-Recurrent Neural Network for Hyperspectral Image Denoising"](https://arxiv.org/abs/2003.04547)

> 🌟 See also the follow up works of QRNN3D:
> - [DPHSIR](https://github.com/Zeqiang-Lai/DPHSIR) - Plug-and-play QRNN3D that solve any HSI restoration task in one model.
> - [HSDT](https://github.com/Zeqiang-Lai/HSDT) - State-of-the-art HSI denoising transformer that follows up 3D paradigam of QRNN3D.
> - [MAN](https://github.com/Zeqiang-Lai/MAN) - Improved QRNN3D with significant performance improvement and less parameters.
>
> 📉 Performance: QRNN3D < DPHSIR < MAN < HSDT

## Highlights

* Our network outperforms all leading-edge methods (2019)
on ICVL dataset in both Gaussian and complex noise cases, as shown below:


<img src="imgs/runtime_gauss.png" height="140px"/> <img src="imgs/runtime_complex.png" height="140px"/>

* We demonstrated our network pretrained on 31-bands natural HSI database (ICVL) can be utilized to recover remotely-sensed HSI (> 100 bands) corrupted by real-world non-Gaussian noise due to terrible atmosphere and water absorptions

<img src="imgs/PaviaU.gif" height="140px"/>  <img src="imgs/Indian_pines.gif" height="140px"/>  <img src="imgs/Urban.gif" height="140px"/> 


## Prerequisites
* Python >=3.5, PyTorch >= 0.4.1
* Requirements: opencv-python, tensorboardX, caffe
* Platforms: Ubuntu 16.04, cuda-8.0


## Quick Start

### 1. Preparing your training/testing datasets

Download ICVL hyperspectral image database from [here](http://icvl.cs.bgu.ac.il/hyperspectral/) (we only need ```.mat``` version)

* The train-test split can be found in ```ICVL_train.txt``` and ```ICVL_test_*.txt```. (Note we split the 101 testing data into two parts for Gaussian and complex denoising respectively.)

#### Training dataset

*Note cafe (via conda install) and lmdb are required to execute the following instructions.*

* Read the function ```create_icvl64_31``` in ```utility/lmdb_data.py``` and follow the instruction comment to define your data/dataset address. 

* Create ICVL training dataset by ```python utility/lmdb_data.py```

#### Testing dataset

*Note matlab is required to execute the following instructions.*

* Read the matlab code of ```matlab/generate_dataset*``` to understand how we generate noisy HSIs.

* Read and modify the matlab code of ```matlab/HSIData.m``` to generate your own testing dataset

### 2. Testing with pretrained models

* Download our pretrained models from [OneDrive](https://1drv.ms/u/s!AqddfvhavTRiijWftKWgLfUgdSaD?e=nHGjIk) and move them to ```checkpoints/qrnn3d/gauss/``` and ```checkpoints/qrnn3d/complex/``` respectively.

* [Blind Gaussian noise removal]:   
```python hsi_test.py -a qrnn3d -p gauss -r -rp checkpoints/qrnn3d/gauss/model_epoch_50_118454.pth```

* [Mixture noise removal]:  
```python hsi_test.py -a qrnn3d -p complex -r -rp checkpoints/qrnn3d/complex/model_epoch_100_159904.pth```

You can also use ```hsi_eval.py``` to evaluate quantitative HSI denoising performance.  

### 3. Training from scratch

* Training a blind Gaussian model firstly by  
```python hsi_denoising_gauss.py -a qrnn3d -p gauss --dataroot (your own dataroot)```

* Using the pretrained Gaussian model as initialization to train a complex model:  
```python hsi_denoising_complex.py -a qrnn3d -p complex --dataroot (your own dataroot) -r -rp checkpoints/qrnn3d/gauss/model_epoch_50_118454.pth --no-ropt```

## Citation
If you find this work useful for your research, please cite: 
```bibtex
@article{wei2020QRNN3D,
  title={3-D Quasi-Recurrent Neural Network for Hyperspectral Image Denoising},
  author={Wei, Kaixuan and Fu, Ying and Huang, Hua},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
```

and follow up works
```bibtex
@article{lai2022dphsir,
    title = {Deep plug-and-play prior for hyperspectral image restoration},
    journal = {Neurocomputing},
    volume = {481},
    pages = {281-293},
    year = {2022},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2022.01.057},
    author = {Zeqiang Lai and Kaixuan Wei and Ying Fu},
}

@inproceedings{lai2023hsdt,
  author = {Lai, Zeqiang and Chenggang, Yan and Fu, Ying},
  title = {Hybrid Spectral Denoising Transformer with Guided Attention},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year = {2023},
}

@article{lai2023mixed,
  title={Mixed Attention Network for Hyperspectral Image Denoising},
  author={Lai, Zeqiang and Fu, Ying},
  journal={arXiv preprint arXiv:2301.11525},
  year={2023}
}
```

## Contact
Please contact me if there is any question (kaixuan.wei at kaust.edu.sa)  
