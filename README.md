# SEED Reimplementation

This is the reimplementatin of the ICLR'20 paper: [Inductive and Usupervised Representation Learning on Graph Structured Objects](https://openreview.net/pdf?id=rkem91rtDB).

## Introduction

This paper proposes a novel unsupervised graph representation learning model named SEED. SEED contains two major steps 1) random walk with WEAVE encoding and 2) encoder and decoder training.

In this implementation, the random walk and WEAVE is pre-obtained and saved as the prior dataset. The encoder and decoder associated with the identity kernel is implemented.

## Implementation details

### The environment details:
* Ubuntu 16.04
* Python 3.5.5
* TensorFlow 1.5.0
* CUDA 9.0
* Cudnn 7

### File structure:
There are only one .py file as a demo for MUTAG dataset. The folder NUTAG_dataset contains the pre-arranged WEAVE encoding with different walk numbers per graph and walk length. Slight parameter modification is required to load different settings.

```
.
├── README.md                          
├── MUTAG_dataset                            
│     ├── MUTAG_sample_100_length_10.mat
│     ├── MUTAG_sample_150_length_10.mat
│     ├── MUTAG_sample_200_length_5.mat
│     ├── MUTAG_sample_200_length_10.mat
│     └── MUTAG_sample_200_length_15.mat
├── representation_MUTAG_sample_200_length_10
└── SEED_MUTAG_demo.py
```

### Run the code
Simply run the python code:
```
python SEED_MUTAG_demo.py
```
It will save the learned graph representations in the folder ./representation_MUTAG_sample_200_length_10 in every 200 iterations. Other classification and visualization approaches (e.g., SVM, NN, and t-SNE) could be used for quality evaluation.
