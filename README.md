# Efficient Split Learning

Welcome to the GitHub repository for Efficient Split Learning (ESL), a novel approach to optimizing distributed training of deep learning models on resource-constrained devices.

Check out the abstract and get early access to our paper: [Link](https://ieeexplore.ieee.org/document/10587192)

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Key Contributions](#key-contributions)
- [Experimental Results](#experimental-Results)
- [Dataset Used](#dataset-used)
- [installation](#installation)
- [Setup Environment](#setup-environment)
- [Usage](#usage)
- [Acknowlegment](#acknowlegment)
- [Citing ESL](#cite-esl)

## Abstract
Distributed training of deep learning models on resource-constrained devices has gained significant interest. Federated Learning (FL) and Split Learning (SL) are two prominent techniques for collaborative training. ESL optimizes these techniques by reducing device computation during parallel model training and minimizing high communication costs due to frequent exchanges of models, data, and gradients. ESL introduces a key-value store for caching intermediate activations and customizing state-of-the-art neural networks for split learning. This approach allows clients to learn personalized models tailored to their specific data distributions.

## Introduction
Deep learning (DL) models require significant memory, computational resources, and energy for the training process. Traditionally, they are trained centrally on a server using data collected from end devices. With the advent of IoT, data privacy concerns arise, making federated learning (FL) a viable solution. However, deploying FL on resource-constrained IoT devices poses challenges due to their limited processing capabilities and communication overheads. ESL addresses these issues by combining split learning (SL) and FL to optimize communication and computation, ensuring efficient training on IoT devices.

## Key Contributions
1. Transfer Learning and Personalization: Adaptation of transfer learning in the SL framework for improved performance.

2. Key-Value Store: Introduction of a server-side key-value store to cache activation values, reducing the need for repetitive data transmission.

3. Customized Layers: Addition of custom layers to the client's backend to handle non-IID data distributions efficiently.

4. Extensive Evaluation: Demonstrated improvements over baseline FL techniques on real-world federated benchmarks for image classification and 3D segmentation.

## Experimental Results
1. Computation Reduction: 1623x reduction for image classification and 23.9x for 3D segmentation on resource-constrained devices.
   
2. Communication Traffic Reduction: 3.92x reduction for image classification and 1.3x for 3D segmentation during training.
   
3. Accuracy Improvement: Improved average accuracy by 35% for image classification and 31% for 3D segmentation compared to baseline FL techniques.

## Dataset Used
We have used multiple datasets from different with diffrent complexities. The details about them are as follows:
   
| **Data Set**      | **No. of Classes** | **Metric Used** | **Task** | **Base Model** | **Pretrained on Dataset** |
|:--------------------------:|:------------------:|:------------------:|:---------------------:|:-------------------:|:------------------------------:|
| **CIFAR-10**        | 10            | Accuracy            | Image Classification                   | MobileNetV3                  | ImageNet                      | 
| **ISIC-2019**      | 8            | Balanced Accuracy        | Medical Image Classification | ResNet-18      | ImageNet         | 
| **KITS19**        | 3            | Dice Score     | 3D-Image Segmentation                  | nnUNet                  | MSD Pancreas                       |  
| **IXI-Tiny** | 2         | Dice Score       | 3D-Image Segmentation                  | 3D UNet               | MSD Spleen                     | 

## Installation
1. Clone the repository and navigate to the project directory:

   ```
   git clone https://github.com/Manisha-IITBH/Efficient-Split-Learning.git --recursive
   
   cd Efficient-Split-Learning
   ```

2. Install the required dependencies:
   
   ```
   pip install -r requirements.txt
   ```

3. update `WANDB_KEY` in config.py

    
## Setup Environment
Activate the Environment using following command:  `source ./.venv/bin/activate`

## Usage
We use a master trainer script that invokes a specific trainer for each dataset. Run the main script to start the training process:

##### CIFAR10 EXAMPLE:

```
python trainer.py -c 10 -bs 64 -tbs 256 -n 80 --client_lr 1e-3 --server_lr 1e-3 --dataset CIFAR10 --seed 42 --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb

python trainer.py -c 10 -bs 64 -tbs 256 -n 80 --client_lr 1e-3 --server_lr 1e-3 --dataset CIFAR10 --seed 42 --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb > text.txt
```

##### KiTS-19 EXAMPLE:

```
python trainer.py -c 6 -bs 4 -tbs 2 -n 30 --client_lr 6e-4 --server_lr 6e-4 --dataset kits19 --seed 42 --model nnunet --split 3 -kv --kv_refresh_rate 5 --kv_factor 2 --wandb
```


###### Options:
```
options:
  -h, --help            show this help message and exit
  -c C, --number_of_clients C
                        Number of Clients (default: 6)
  -bs B, --batch_size B
                        Batch size (default: 2)
  -tbs TB, --test_batch_size TB
                        Input batch size for testing (default: 2)
  -n N, --epochs N      Total number of epochs to train (default: 10)
  --client_lr LR        Client-side learning rate (default: 0.001)
  --server_lr serverLR  Server-side learning rate (default: 0.001)
  --dataset DATASET     States dataset to be used (default: kits19)
  --seed SEED           Random seed (default: 42)
  --model MODEL         Model you would like to train (default: nnunet)
  --split SPLIT         The model split version to use (default: 1)
  -kv, --use_key_value_store
                        use key value store for faster training (default: False)
  --kv_factor KV_FACTOR
                        populate key value store kv_factor times (default: 1)
  --kv_refresh_rate KV_REFRESH_RATE
                        refresh key-value store every kv_refresh_rate epochs, 0 =
                        disable refresing (default: 5)
  --wandb               Enable wandb logging (default: False)
  --pretrained          Model is pretrained/not, DEFAULT True, No change required
                        (default: True)
  --personalize         Enable client personalization (default: False)
  --pool                create a single client with all the data, trained in split
                        learning mode, overrides number_of_clients (default: False)
  --dynamic             Use dynamic transforms, transforms will be applied to the
                        server-side kv-store every epoch (default: False)
  --p_epoch P_EPOCH     Epoch at which personalisation phase will start (default: 50)
  --offload_only        USE SERVER ONLY FOR OFFLOADING, CURRENTLY ONLY IMPLEMENTED FOR
                        IXI-TINY (default: False)
```

---

## Acknowledgment
This project makes use of the [PFSL](https://paperswithcode.com/paper/pfsl-personalized-fair-split-learning-with) framework. If you use this project in your research, please consider citing:

```bibtex
@software{Manas_Wadhwa_and_Gagan_Gupta_and_Ashutosh_Sahu_and_Rahul_Saini_and_Vidhi_Mittal_PFSL_2023,
author = {Manas Wadhwa and Gagan Gupta and Ashutosh Sahu and Rahul Saini and Vidhi Mittal},
month = {2},
title = {{PFSL}},
url = {https://github.com/mnswdhw/PFSL},
version = {1.0.0},
year = {2023} 
}
```

## Citing ESL
If you have found the code and/or its ideas useful, please consider citing:

```bibtex
@ARTICLE{10587192,
  author={Chawla, Manisha and Gupta, Gagan Raj and Gaddam, Shreyas and Wadhwa, Manas},
  journal={IEEE Internet of Things Journal}, 
  title={Beyond Federated Learning for IoT: Efficient Split Learning With Caching & Model Customization}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Computational modeling;Training;Servers;Internet of Things;Data models;Quantization (signal);Convergence;Communication Reduction;Federated Learning;IoT;Key-Value Store;Personalization;Resource-Constrained Devices;Split Learning},
  doi={10.1109/JIOT.2024.3424660}}
```
