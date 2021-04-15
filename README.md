# PerceptualLossNetwork
This repository is for CS577 (Deep Learning) @ IIT
The work in this repository is based on the paper:

[Perceptual Losses for Real-Time Style Transfer and Super Resolution](https://arxiv.org/pdf/1603.08155v1.pdf)

```
@article{
title={Perceptual Losses for Real-Time Style Transfer and Super Resolution},
author={Justin Johnson, Alexandre Alahi, Li Fei-Fei},
publisher={Dept. of Computer Science, Stanford University},
year={2016}
}
```
This implementation was inspired from the offical [pytorch example](https://github.com/pytorch/examples/tree/master/fast_neural_style) from facebook research:


## Installation
### MacOS (CPU)
- Install conda 
- Set env by either:
    - Create conda enviornment: `conda create --name <env> --file macOS_local_req.txt`
    - Install into existing conda environment: `conda install -n <env_name> macOS_local_req.txt`

## Perceptual Loss
### VGG16 Architecture:
![VGG16 Architecture](vgg16ARCH.png)
