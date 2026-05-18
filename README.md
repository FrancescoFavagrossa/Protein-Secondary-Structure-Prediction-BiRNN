# Protein Secondary Structure Prediction using Deep Learning

This repository contains a deep learning project for **protein secondary structure prediction** from amino acid sequences.

The goal is to predict the secondary structure of each amino acid in a protein sequence using recurrent neural network architectures. The project focuses on the simplified **Q3 classification task**, where each residue is classified as one of three secondary structure classes:

- **C**: Coil
- **H**: Alpha-helix
- **E**: Beta-sheet

## Project Overview

Protein structure prediction is a central problem in computational biology and bioinformatics. While full 3D structure prediction is a highly complex task, secondary structure prediction represents an important intermediate step.

In this project, several sequence-based deep learning models are implemented and compared:

- LSTM
- GRU
- Bidirectional LSTM
- Bidirectional GRU

The models are trained to predict a secondary structure label for each amino acid in the input sequence.

## Dataset

The dataset used in this project is derived from the **RCSB Protein Data Bank (PDB)** and obtained from Kaggle.

Dataset characteristics:

- Protein sequences with structural annotations
- PISCES-filtered subset
- 30% sequence identity cutoff
- 2.5 Å resolution threshold
- DSSP-based secondary structure annotations
- Q3 labels: Coil, Helix, Sheet

Dataset link:

https://www.kaggle.com/datasets/kirkdco/protein-secondary-structure-2022

## Data Preprocessing

The preprocessing pipeline includes:

1. Filtering protein sequences with length between 30 and 500 amino acids
2. Removing sequences containing non-standard amino acids
3. Encoding amino acids as integers
4. Padding sequences to a maximum length of 500
5. Encoding secondary structure labels into categorical format
6. Splitting the dataset into training and test sets

The 20 standard amino acids are mapped to integer values, while padding is represented with zero.


## Recurrent Neural Networks and Gating Mechanisms

Recurrent Neural Networks (RNNs) are neural architectures designed to process sequential data. In this project, protein sequences are treated as ordered sequences of amino acids, where the prediction for each residue may depend on both previous and following residues.

A standard RNN updates its hidden state at each time step as:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

where:

- $x_t$ is the input at time step $t$
- $h_t$ is the hidden state at time step $t$
- $h_{t-1}$ is the previous hidden state
- $W_x$ and $W_h$ are weight matrices
- $b$ is a bias term

However, standard RNNs often suffer from the **vanishing gradient problem**, which makes it difficult to learn long-range dependencies. This is particularly important in protein sequences, where the structure of one amino acid can depend on residues that are far away in the sequence.

To address this issue, gated recurrent architectures such as **LSTM** and **GRU** introduce gates that control how information is stored, updated, and forgotten.

### LSTM Gates

Long Short-Term Memory networks use a memory cell $c_t$ and three main gates: the forget gate, input gate, and output gate.

#### Forget Gate

The forget gate decides how much information from the previous cell state $c_{t-1}$ should be kept:

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

where $f_t$ contains values between 0 and 1. A value close to 0 means that the information is mostly forgotten, while a value close to 1 means that it is mostly preserved.

#### Input Gate

The input gate decides how much new information should be added to the cell state:

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

A candidate cell state is also computed:

$$
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
$$

The cell state is then updated as:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

where $\odot$ denotes element-wise multiplication.

#### Output Gate

The output gate controls how much of the cell state is used to produce the hidden state:

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

The LSTM architecture is useful because the cell state acts as a memory channel, allowing relevant information to be carried across many time steps.

### GRU Gates

Gated Recurrent Units are a simplified alternative to LSTMs. GRUs do not use a separate cell state. Instead, they rely on two main gates: the update gate and the reset gate.

#### Update Gate

The update gate controls how much of the previous hidden state should be retained:

$$
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)
$$

A high value of $z_t$ means that more past information is preserved.

#### Reset Gate

The reset gate determines how much of the previous hidden state should be ignored when computing the candidate hidden state:

$$
r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)
$$

The candidate hidden state is computed as:

$$
\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)
$$

The final hidden state is then updated as:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

GRUs are often computationally lighter than LSTMs because they use fewer gates and parameters. In this project, the Bidirectional GRU achieved the best performance, suggesting that it was able to capture useful sequence dependencies while remaining relatively efficient.

### Why Bidirectional RNNs Are Useful for Protein Sequences

In a standard recurrent model, the hidden state at time $t$ mainly depends on the previous residues:

$$
h_t = f(x_t, h_{t-1})
$$

In a bidirectional recurrent model, the sequence is processed in both directions:

$$
\overrightarrow{h_t} = f(x_t, \overrightarrow{h_{t-1}})
$$

$$
\overleftarrow{h_t} = f(x_t, \overleftarrow{h_{t+1}})
$$

The final representation is usually obtained by combining the forward and backward hidden states:

$$
h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]
$$

This is particularly useful for protein secondary structure prediction because the structural class of an amino acid depends not only on previous residues, but also on following residues.


## Models

### LSTM

The LSTM model uses:

- Embedding layer
- LSTM layer with 64 units
- Dropout rate of 0.2
- TimeDistributed dense output layer with softmax activation

### GRU

The GRU model follows the same structure as the LSTM model, replacing the LSTM layer with a GRU layer.

### Bidirectional LSTM

The Bidirectional LSTM processes each protein sequence in both directions, allowing the model to use information from both previous and following amino acids.

This is particularly useful because the secondary structure of an amino acid can depend on residues located both before and after it in the sequence.

### Bidirectional GRU

The Bidirectional GRU follows the same idea as the Bidirectional LSTM, but uses GRU units instead of LSTM units.

## Results

| Model | Test Accuracy | Training Time |
|---|---:|---:|
| LSTM | 59.56% | ~66 min |
| GRU | 59.32% | ~66 min |
| Bidirectional LSTM | 69.23% | ~120 min |
| Bidirectional GRU | 69.38% | ~119 min |

The bidirectional models clearly outperform the unidirectional ones. This result is consistent with the biological nature of the problem, since the secondary structure of a residue depends on both upstream and downstream amino acids.

The best-performing model is the **Bidirectional GRU**, with a test accuracy of approximately **69.4%**.

## Main Takeaways

- Protein secondary structure prediction can be approached as a sequence labeling problem.
- Recurrent neural networks are suitable for modeling amino acid sequences.
- Bidirectional architectures perform better than unidirectional ones because they use both past and future sequence context.
- The Bidirectional GRU achieved the best trade-off between accuracy and computational cost in this project.

## Future Improvements

Possible extensions of this project include:

- Hyperparameter tuning
- Use of convolutional neural networks
- Transformer-based architectures
- Comparison with pretrained protein language models
- Evaluation with precision, recall, F1-score, and confusion matrices
- Extension from Q3 to Q8 secondary structure classification

## Author

**Francesco Favagrossa**
