#TensorGCN

The implementation of TensorGCN in paper:

Liu X, You X, Zhang X, et al. Tensor graph convolutional networks for text classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(05): 8409-8416.


# Require

Python 3.6

Tensorflow >= 1.11.0


# Reproduing Results

####1. Build three graphs

Run TGCN_2layers/build_graph_tgcn.py

####2. Training

Run TGCN_2layers/train.py


# Example input data

1. `/data_tgcn/mr/build_train/mr.clean.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data_tgcn/mr/build_train/mr.txt` contains raw text of each document.

3. `/data_tgcn/mr/stanford/mr_pair_stan.pkl` contains all syntactic relationship word pairs for the dataset.

4. `/data_tgcn/mr/build_train/mr_semantic_0.05.pkl` contains all semantic relationship word pairs for the dataset.
