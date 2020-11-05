# Hyperspectral Image Denoising - A Comprehensive Benchmark 

An **unified interface** for benchmarking HSI denoising algorithms on various datasets under different noise settings. 

## Quick Start

* Download the library of HSI denoising algorithms from [OneDrive](https://1drv.ms/u/s!AqddfvhavTRii3kvriy6C14ub-SH?e=wdNnRf) and put the ```lib``` directory in the ```matlab``` folder.

* Type ```addpath(genpath('lib'));``` in matlab command line; now you can use any algorithms with an unified interface defined in ```demo_fun.m```. For instance, you can use ```BM4D``` to denoise a hyperspectral image via ```demo_fun(noisy_hsi, sigma_ratio, 'BM4D')```

* Benchmark your selected algorithm on the whole dataset by ```Main_Gauss/Complex/Real.m```; Get the quantitative results via ```Result_Gauss/Complex.m```

## Citation

The bibtex of the collected algorithms can be found in ```benchmarks.bib```

## Acknowledgments
* The code of different HSI denoising algorithms is collected online.
* Special thanks to these authors for making their source code publicly available. 


## Contributions

Adding your own algorithms is very easy: Simply put your source code in ```lib``` directory, then write your algorithm interface in ```demo_fun.m```. 
