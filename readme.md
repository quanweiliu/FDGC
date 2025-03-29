# A Fast Dynamic Graph Convolutional Network and CNN Parallel Network for Hyperspectral Image Classification

This example implements the paper in review [A Fast Dynamic Graph Convolutional Network and CNN Parallel Network for Hyperspectral Image Classification]

## Run
If you want to run this code, just put your data in the Datasets folder and change a few paths.
- path 1: main.py:path-config.
- path 2: data_reader.py: add or change to your data path, just in Folder path.
- config.yaml: your Folder path and dataset name, your weight and result store path.

then:

> python main.py 

## Installation
This project is implemented with Pytorch and has been tested on version 
- Pytorch               1.7, 
- numpy                 1.21.4
- matplotlib            3.3.3 
- scikit-learn          0.23.2


## Citation
Please kindly cite the papers [A Fast Dynamic Graph Convolutional Network and CNN Parallel Network for Hyperspectral Image Classification](https://ieeexplore.ieee.org/abstract/document/9785802) if this code is useful and helpful for your research.

```
@article{liu2022fast,
  title={A fast dynamic graph convolutional network and CNN parallel network for hyperspectral image classification},
  author={Liu, Quanwei and Dong, Yanni and Zhang, Yuxiang and Luo, Hui},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--15},
  year={2022},
  publisher={IEEE}
}
```
