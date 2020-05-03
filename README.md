# Semantic Pyramid for Image Generation
**Unofficel** [PyTorch](https://pytorch.org/) implementation of the paper [Semantic Pyramid for Image Generation](https://arxiv.org/pdf/2003.06221.pdf) by Assaf Shocher & Yossi Gandelsman.

![Results](figures/paper_results_overview.png "Paper results")
[Source](https://arxiv.org/pdf/2003.06221.pdf). Proposed results of the paper.

## Model architecture

![Architecture](figures/paper_architecture.png "Paper architecture")
[Source](https://arxiv.org/pdf/2003.06221.pdf)

The full architecture consists of three parts. First, the object recognition model which is implemented as a
pre-trained VGG 16 network. Secondly, the residual generator network which is partly based on the generator architecture 
of the [SAGAN](https://arxiv.org/pdf/1805.08318.pdf).
And thirdly, the residual discriminator network which is also based on the 
[SAGAN](https://arxiv.org/pdf/1805.08318.pdf).

## Dataset
To download and extract the [places365](http://places2.csail.mit.edu/download.html) dataset from the official website
run the following script
```
sh download_places365.sh
```
## Pre-trained VGG 16 model
To download and convert the VGG 16 model pre-trained on the places dataset run the following command
```
sh download_pretrained_vgg16.sh
```
This script downloads the official pre-trained caffe models from the
[places365 repository](https://github.com/CSAILVision/places365). Afterwards, the caffe model gets converted with the
help of the [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch) repo created by 
[Vadim Kantorov](https://github.com/vadimkantorov).

## Usage
To train or test the GAN simply run the main script `python main.py`. This main script takes multiple arguments.

Argument | Default value | Info
--- | --- | ---
`--train` | 1 (True) | Flag to perform training
`--test` | 1 (True) | Flag to perform testing
`--batch_size` | 20 | Batch size to be utilized
`--lr` | 1e-04 | Learning rate to use
`--channel_factor` | 1 | Channel factor adopts the number of channels utilized in the U-Net
`--device` | 'cuda' | Device to use (cuda recommended)
`--gpus_to_use` | '0' | Indexes of the GPUs to be use
`--use_data_parallel` | 0 (False) | Use multiple GPUs
`--load_generator_network` | None | Path of the generator network to be loaded (.pt)
`--load_discriminator_network` | None | Path of the discriminator network to be loaded (.pt)
`--load_pretrained_vgg16` | 'pre_trained_models/vgg_places_365.pt' | Path of the pre-trained VGG 16 to be loaded (.pt)
`--path_to_places365` | 'places365_standard' | Path to places365 dataset
`--epochs` | 100 | Epochs to perform while training

While training logs and models gets save automatically to the `saved_data` file. Inference plots also gets saved in the
same folder.

## Results
With limited compute budget I was not able to reproduce the results form the paper. The plot, shown below, was after approximately 24h of training on a single Nvidia Tesla V1000. After 24h the whole performance dropped again. However, due to the limited computing power, I was only able to train 48h.

![plot](figures/600000.png")
