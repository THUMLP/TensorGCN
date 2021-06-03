#TensorGCN

The implementation of [TensorGCN](https://arxiv.org/pdf/2001.05313.pdf) in paper:

Liu X, You X, Zhang X, et al. Tensor graph convolutional networks for text classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(05): 8409-8416.


# Require

Python 3.6

Tensorflow >= 1.11.0


# Reproduing Results

####1. Build three graphs

Run TGCN1_2layers/build_graph_tgcn.py

####2. Training

Run TGCN1_2layers/train.py


# Example input data

1. `/data_tgcn/mr/build_train/mr.clean.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data_tgcn/mr/build_train/mr.txt` contains raw text of each document.

3. `/data_tgcn/mr/stanford/mr_pair_stan.pkl` contains all syntactic relationship word pairs for the dataset.

4. `/data_tgcn/mr/build_train/mr_semantic_0.05.pkl` contains all semantic relationship word pairs for the dataset.


#Semantic-based graph
we propose a LSTM-based method to construct a semantic-based graph from text documents. There are three main steps:
- Step 1: Train a LSTM on the training data of the given task (e.g. text classification here).
- Step 2: Get semantic features/embeddings with LSTM for all words in each document/sentence of the corpus.
- Step 3: Calculate word-word edge weights based on word semantic embeddings over the corpus.The calculation formula can be found in formula (3) in the paper.


#Syntactic-based graph
- Step 1: We utilize stanford CoreNLP parser to extract dependency between words. You can learn how to use the toolkit through [this website](https://www.pianshen.com/article/8433287443/)
- Step 2: Get syntactic relationship word pairs for the dataset by :
  Run TGCN1_2layers/get_syntactic_relationship.py. 