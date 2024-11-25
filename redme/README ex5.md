# Implementation and evaluation of Graph Transformers with different positional encodings

This projects focuses on testing a graph transformer model on the Cora and the Peptides-func datasets. We will use laplatian positional encodings and random walk structural embeddings to capture the edge distribution of the graphs.

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Tasks Overview](#tasks-overview)
    - [Task 1: Laplacian Positional Encodings (LapPE)](#task-1-laplacian-positional-encodings-lappe)
    - [Task 2: Combining LapPE with SignNet](#task-2-combining-lappe-with-signnet)
    - [Task 3: Random Walk Structural Embeddings (RWSE)](#task-3-random-walk-structural-embeddings-rwse)
    - [Task 4: Pure Graph Transformer](#task-4-pure-graph-transformer)
    - [Task 5: Performance Comparison of LapPE and RWSE](#task-5-performance-comparison-of-lappe-and-rwse)
4. [Datasets](#datasets)
5. [Usage](#usage)
6. [Results and Observations](#results-and-observations)
7. [References](#references)

---

## Tasks Overview

### Task 1: Laplacian Positional Encodings (LapPE)
- **Goal**: Compute Laplacian eigenvectors for a graph using the Laplacian matrix \( L = D - A \).
- **Steps**:
  1. Define a `compute_lap_pe()` function for calculating LapPE.
  2. Test the function on a custom dataset.
- **Outputs**: Laplacian positional encodings for the nodes in the graph.

---

### Task 2: Combining LapPE with SignNet
- **Goal**: Make LapPE embeddings sign-invariant using a SignNet-based model.
- **Steps**:
  1. Define a `SignNet` class and the `compute_lap_pe_with_signnet()` function.
  2. Test the implementation on a custom graph.
- **Outputs**: Sign-invariant Laplacian positional encodings.

---

### Task 3: Random Walk Structural Embeddings (RWSE)
- **Goal**: Compute RWSE by counting closed walks of a given length \( k \) in a graph.
- **Steps**:
  1. Define the `compute_rwse()` function.
  2. Test it using the *KarateClub* dataset from `torch_geometric.datasets`.
- **Outputs**: Structural embeddings capturing node-level random walk statistics.

---

### Task 4: Pure Graph Transformer
- **Goal**: Implement a graph transformer model to process graphs using global attention.
- **Steps**:
  1. Define a `GraphTransformer` class with a `GraphTransformerEncoderLayer`.
  2. Implement a `GCN` class for baseline comparison.
  3. Define training, testing, and evaluation functions.
  4. Test the model on the *Cora* dataset, recording:
      - Training loss and accuracy
      - Validation accuracy
      - Test accuracy for each epoch
- **Outputs**: A trained graph transformer with accuracy metrics.

---

### Task 5: Performance Comparison of LapPE and RWSE
- **Goal**: Compare the impact of LapPE and RWSE embeddings on model performance.
- **Steps**:
  1. Define a `run_experiments()` function to test various configurations:
      - Baseline (no embeddings)
      - LapPE only
      - RWSE only
      - LapPE + RWSE
  2. Measure performance for each configuration using the metrics from Task 4.
- **Outputs**: Comparative results showcasing the effectiveness of different embeddings.
