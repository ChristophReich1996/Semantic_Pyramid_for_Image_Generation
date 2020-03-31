# Semantic Pyramid for Image Generation
## WORK IN PROGRESS...
**Unofficel** [PyTorch](https://pytorch.org/) implementation of the paper [Semantic Pyramid for Image Generation](https://arxiv.org/pdf/2003.06221.pdf) by Assaf Shocher & Yossi Gandelsman.

![Results](figures/paper_results_overview.png "Paper results")
[Source](https://arxiv.org/pdf/2003.06221.pdf). Proposed results of the paper.

## Model architecture

## Dataset
To download and extract the [places365](http://places2.csail.mit.edu/download.html) dataset from the official website
run the following script
```
sh download_places365.sh
```
## Pre-trained VGG 16 model
To download and convert the VGG 16 model pre-trained on the places dataset run the following command
```
sh download_pretrained_vgg16_.sh
```
This script downloads the official pre-trained caffe models from the
[places365 repository](https://github.com/CSAILVision/places365). Afterwards, the caffe model gets converted with the
help of the [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch) repo created by 
[Vadim Kantorov](https://github.com/vadimkantorov).

## Usage
To train or test the GAN simply run the main script `python main.py`. This main script takes multiple arguments.
