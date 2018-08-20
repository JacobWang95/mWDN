Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis
============================
This repository is code release for time series classification task of our recent KDD2018 [paper](https://arxiv.org/pdf/1806.08946.pdf).

The code is based on Tensorflow and other basic python packages.

We only tested the code on Ubuntu 16.04 with python 2.7 and TensorFLow 1.2, a GPU with at least 4GB memory is recommanded.

##Usage
Before running the code, modify the line 65 of train.py to let the DATA_ROOT point to the path of UCR dataset on your machine.

To start a demo on training on yoga data in UCR dataset, simply run the following command:
```
python train.py --gpu 0 --log_dir log_demo_yoga
```
The code is designed to automatically save the best model parameters.

You may try other args by adding them to the command, for details please refer to:
```
python train.py --help

```
