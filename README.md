Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis
============================
This repository is code release for time series classification task of our recent KDD2018 [paper](https://arxiv.org/pdf/1806.08946.pdf).

The code is based on Tensorflow and other basic python packages.

We only tested the code on Ubuntu 16.04 with python 2.7 and TensorFLow 1.2, a GPU with at least 4GB memory is recommanded.

# Usage

Before running the code, modify the line 65 of train.py to let the DATA_ROOT points to the path to UCR dataset on your machine.

To start a demo on training yoga data of UCR dataset, simply run the following command:
```
python train.py --gpu 0 --log_dir log_demo_yoga
```
The code is designed to automatically save the best model parameters.

You may try other args by adding them to the command, for details please refer to:
```
python train.py --help
```
# Citing

If you find our work is helpful for your research, please kindly consider citing our paper as well.

```latex
@inproceedings{Wang:2018:MWD:3219819.3220060,
 author = {Wang, Jingyuan and Wang, Ze and Li, Jianfeng and Wu, Junjie},
 title = {Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis},
 booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \&\#38; Data Mining},
 series = {KDD '18},
 year = {2018},
 isbn = {978-1-4503-5552-0},
 location = {London, United Kingdom},
 pages = {2437--2446},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3219819.3220060},
 doi = {10.1145/3219819.3220060},
 acmid = {3220060},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {epidemic propagation, intracity epidemic control and prevention, metapopulation, network inference},
}
```
