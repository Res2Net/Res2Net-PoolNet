# Res2Net for Salient Object Detection using PoolNet

## Introduction
This repo uses [*PoolNet* (cvpr19)](https://arxiv.org/abs/1904.09569) as the baseline method for Salient Object Detection . 

[Res2Net](https://github.com/gasvn/Res2Net) is a powerful backbone architecture that can be easily implemented into state-of-the-art models by replacing the bottleneck with Res2Net module.
More detail can be found on [ "Res2Net: A New Multi-scale Backbone Architecture"](https://arxiv.org/pdf/1904.01169.pdf)

## Performance

### Results on salient object detection datasets **without** joint training with edge. Models are trained using DUTS-TR.

| Backbone     | ECSSD        | PASCAL-S      | DUT-O         | HKU-IS         | SOD             | DUTS-TE        |
|--------------|--------------|---------------|---------------|----------------|-----------------|----------------|
|    -         | MaxF & MAE   | MaxF & MAE    | MaxF & MAE    | MaxF & MAE     | MaxF & MAE      | MaxF & MAE     |
| vgg          |0.936 & 0.047 | 0.857 & 0.078 | 0.817 & 0.058 |  0.928 & 0.035 |   0.859 & 0.115 |  0.876 & 0.043 |
| resnet50     |0.940 & 0.042 | 0.863 & 0.075 | 0.830 & 0.055 |  0.934 & 0.032 |   0.867 & 0.100 |  0.886 & 0.040 |
| **res2net50**|0.947 & 0.036 | 0.871 & 0.070 | 0.837 & 0.052 |  0.936 & 0.031 |   0.885 & 0.096 |  0.892 & 0.037 |



## Evaluation

You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).

## Todo
We will merge this repo into the official repo of PoolNet soon.
We only modify the normalization of inputs of the PoolNet as follows:
```
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
```
## Usage

### Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)

### 1. Clone the repository

```shell
https://github.com/gasvn/Res2Net-PoolNet.git
cd Res2Net_PoolNet/
```

### 2. Download the datasets

Download the following datasets and unzip them into `data` folder.

* [MSRA-B and HKU-IS](https://drive.google.com/open?id=14RA-qr7JxU6iljLv6PbWUCQG0AJsEgmd) dataset. The .lst file for training is `data/msrab_hkuis/msrab_hkuis_train_no_small.lst`.
* [DUTS](https://drive.google.com/open?id=1immMDAPC9Eb2KCtGi6AdfvXvQJnSkHHo) dataset. The .lst file for training is `data/DUTS/DUTS-TR/train_pair.lst`.
* [BSDS-PASCAL](https://drive.google.com/open?id=1qx8eyDNAewAAc6hlYHx3B9LXvEGSIqQp) dataset. The .lst file for training is `./data/HED-BSDS_PASCAL/bsds_pascal_train_pair_r_val_r_small.lst`.
* [Datasets for testing](https://drive.google.com/open?id=1eB-59cMrYnhmMrz7hLWQ7mIssRaD-f4o).

### 3. Download the pre-trained models for backbone

Download the pretrained models of Res2Net50 from [Res2Net](https://github.com/gasvn/Res2Net) .
Set the path to pretrain model of Res2Net in `main.py`  (line 55)
```
res2net_path = '/home/shgao/.torch/models/res2net50_26w_4s-06e79181.pth'
```
### 4. Train

1. Set the `--train_root` and `--train_list` path in `train_res2net.sh` correctly.

2. We demo using Res2Net-50 as network backbone and train with a initial lr of 5e-5 for 24 epoches, which is divided by 10 after 15 epochs.
```shell
./train_res2net.sh
```
3. We demo joint training with edge using Res2Net-50 as network backbone and train with a initial lr of 5e-5 for 11 epoches, which is divided by 10 after 8 epochs. Each epoch runs for 30000 iters.
```shell
./joint_train_res2net.sh
```
4. After training the result model will be stored under `results/run-*` folder.

### 5. Test

For single dataset testing: `*` changes accordingly and `--sal_mode` indicates different datasets (details can be found in `main.py`)
```shell
python main.py --mode='test' --model='results/run-*/models/final.pth' --test_fold='results/run-*-sal-e' --sal_mode='e' --arch res2net
```
For all datasets testing used in our paper: `0` indicates the gpu ID to use
```shell
./forward.sh 0 main.py results/run-*
```
For joint training, to get salient object detection results use
```shell
./forward.sh 0 joint_main.py results/run-*
```
to get edge detection results use
```shell
./forward_edge.sh 0 joint_main.py results/run-*
```

All results saliency maps will be stored under `results/run-*-sal-*` folders in .png formats.


### 6. Pre-trained models

The pretrained models for SOD using Res2Net is now available on [ONEDRIVE](https://1drv.ms/u/s!AkxDDnOtroRPe43-1JjD304ecvU?e=Y7qCHN).

Note：

1. only support `bath_size=1`
2. Except for the backbone we do not use BN layer.




## Applications
Other applications such as Classification, Instance segmentation, Object detection, Segmantic segmentation, pose estimation, Class activation map can be found on https://mmcheng.net/res2net/ and https://github.com/gasvn/Res2Net .

## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758}, 
}
@inproceedings{Liu2019PoolSal,
  title={A Simple Pooling-Based Design for Real-Time Salient Object Detection},
  author={Jiang-Jiang Liu and Qibin Hou and Ming-Ming Cheng and Jiashi Feng and Jianmin Jiang},
  booktitle={IEEE CVPR},
  year={2019},
}
```
## Acknowledge
The code for salient object detection is partly borrowed from [A Simple Pooling-Based Design for Real-Time Salient Object Detection](https://github.com/backseason/PoolNet).
