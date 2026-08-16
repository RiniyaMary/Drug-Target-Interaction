# Multi-modal Feature Fusion for Drug-Target Interaction Prediction

## Overview

Drug-Target Interaction (DTI) prediction is an important task in computational drug discovery and drug repurposing. This project presents a multi-modal deep learning framework that combines molecular graph representations, chemical semantic embeddings, and protein sequence features for predicting drug-target interactions.

## Objectives

- Predict interactions between drug molecules and protein targets.
- Capture structural information from molecular graphs.
- Incorporate chemical semantic information using Mol2Vec.
- Extract meaningful features from protein sequences.
- Learn interaction-specific relationships between drug and protein representations.

## Proposed Framework

### Drug Representation

- Molecular graph representation
- Graph Convolutional Network (GCN)
- Mol2Vec chemical semantic embeddings

### Protein Representation

- Protein sequence representation
- Attention-based convolution module (ACmix)
- Captures local and global sequence dependencies

### Interaction and Prediction

- Bi-Intention Cross-Attention
- Feature aggregation
- MLP classifier
- Interaction probability prediction

## Dataset

The model was evaluated using the DrugBank dataset.

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Mol2Vec
- Graph Neural Networks
- Deep Learning
- Jupyter Notebook
- Git

