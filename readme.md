<h1 align="center">
Could SNN Beat ANN on Image Synthesis
</h1>
<p align="center">
Project of AI3610 Brain-inspired Intelligence, Shanghai Jiao Tong University.
</p>


## Introduction


The contributions of this project are as follows:
- We compared various types of Generators and Discriminators in the GAN framework, and found that the SNN-based Generator


## Installation

### Getting start

Clone our repository:

```shell
$ git clone --recursive https://github.com/SylvanSun/Auto-SNN-for-ICG.git
$ cd Auto-SNN-for-ICG
```

Set up a new environment:

```shell
$ conda env create -f environment.yaml
$ conda activate snngan
```

### Download assets and process data

We use default datasets as provided by pytorch. Simply run our scripts for the standard training procedure, the datasets will be downloaded automatically.

The folder `data/` should look like this:

```
data
├── cifar-10-batches-py/
│   └── ...
└── MNIST/
    └── ...
```


## Explanation for the scripts
We support 4 types of Generators and 2 types of Discriminators for now.To train the models, simply run the script with the corresponding arguments as shown below:
```shell
$ python dcgan.py --gen [Generator type] --dis [Discriminator type]
```
The generators should be selected from 
```shell
["ann", "front", "mid", "back"]
```
and the discriminators should be selected from
```shell
["ann", "snn"]
```
The images will be generated every epoch and saved in the folder `images/` with corresponding generator and discriminator. The models will be saved in the folder `asset/model_saved/`.


## For TAs


## Acknowledgement




