# TGRNet: A Table Graph Reconstruction Network for Table Structure Recognition
> Xue, Wenyuan, et al. "TGRNet: A Table Graph Reconstruction Network for Table Structure Recognition." arXiv preprint arXiv:2106.10598 (2021).

This work has been accepted for presentation at ICCV2021. The preview version has released at arXiv.org (<https://arxiv.org/abs/2106.10598>).

## Abstract
A table arranging data in rows and columns is a very effective data structure, which has been widely used in business and scientific research. Considering large-scale tabular data in online and offline documents, automatic table recognition has attracted increasing attention from the document analysis community. Though human can easily understand the structure of tables, it remains a challenge for machines to understand that, especially due to a variety of different table layouts and styles. Existing methods usually model a table as either the markup sequence or the adjacency matrix between different table cells, failing to address the importance of the logical location of table cells, e.g., a cell is located in the first row and the second column of the table. In this paper, we reformulate the problem of table structure recognition as the table graph reconstruction, and propose an end-to-end trainable table graph reconstruction network (TGRNet) for table structure recognition. Specifically, the proposed method has two main branches, a cell detection branch and a cell logical location branch, to jointly predict the spatial location and the logical location of different cells. Experimental results on three popular table recognition datasets and a new dataset with table graph annotations (TableGraph-350K) demonstrate the effectiveness of the proposed TGRNet for table structure recognition.

# Getting Started
## Requirements
Create the environment from the environment.yml file `conda env create --file environment.yml` or install the software needed in your environment independently. If you meet some problems when installing PyTorch Geometric, please follow the official installation indroduction (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
```
dependencies:
  - python==3.7.0
  - pip==20.2.4
  - pip:
    - dominate==2.5.1
    - imageio==2.8.0
    - networkx==2.3
    - numpy==1.18.2
    - opencv-python==4.4.0.46
    - pandas==1.0.3
    - pillow==7.1.1
    - torchfile==0.1.0
    - tqdm==4.45.0
    - visdom==0.1.8.9
    - Polygon3==3.0.8
```
PyTorch Installation
```
# CUDA 10.2
pip install torch==1.5.0 torchvision==0.6.0
# CUDA 10.1
pip install torch==1.5.0+CU101 torchvision==0.6.0+CU101 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 9.2
pip install torch==1.5.0+CU92 torchvision==0.6.0+CU92 -f https://download.pytorch.org/whl/torch_stable.html
```
PyTorch Geometric Installation
```
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install torch-sparse==0.6.3 -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
pip install torch-geometric
```
where ${CUDA} should be replaced by your specific CUDA version (cu92, cu101, cu102).
## Datasets Preparation
- Download datasets from [Google Dive](https://drive.google.com/file/d/19STySr6EYlm1cAbdZIgYR4YbJteGFkQl/view?usp=sharing) or [Alibaba Cloud](https://wenyuancloud.oss-cn-beijing.aliyuncs.com/data/cvpr/datasets.tar.gz).
- Put datasets.tar.gz in "./datasets/" and extract it.
```
cd ./datasets
tar -zxvf datasets.tar.gz
## The './datasets/' folder should look like:
- datasets/
  - cmdd/
  - icdar13table/
  - icdar19_ctdar/
  - tablegraph24k/
```

## Pretrained Models Preparation
- Download pretrained models from [Google Dive](https://drive.google.com/file/d/1qjFGdph3Y_s9sio9ngk6wQEAWduuneIm/view?usp=sharing) or [Alibaba Cloud](https://wenyuancloud.oss-cn-beijing.aliyuncs.com/data/cvpr/checkpoints.tar.gz).
- Put checkpoints.tar.gz in "./checkpoints/" and extract it.
```
cd ./checkpoints
tar -zxvf checkpoints.tar.gz
## The './checkpoints/' folder should look like:
- checkpoints/
  - cmdd_overall/
  - icdar13table_overall/
  - icdar19_lloc/
  - tablegraph24k_overall/
```

## Test
We have prepared scripts for test and you can just run them.
```
- test_cmdd.sh
- test_icdar13table.sh
- test_tablegraph-24k.sh
- test_icdar19ctdar.sh
```
## Train
Todo

