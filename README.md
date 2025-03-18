

# Struct-KGS2S: Structural Context-based Seq2Seq Model for Link Prediction in Knowledge Graphs

This repository contains the code, data, and resources for Struct-KGS2S, a novel Seq2Seq model leveraging structural context to enhance link prediction performance in Knowledge Graphs. This project explores the use of structured contextual embeddings to improve the model's ability to predict relationships between entities, enhancing applications in knowledge-based AI systems.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training & Evaluation](#training--evaluation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Struct-KGS2S is designed to address the challenges in link prediction within Knowledge Graphs by incorporating structural information to provide context. Traditional link prediction models often lack the capability to fully leverage graph structure; Struct-KGS2S aims to bridge this gap using a Seq2Seq approach. The primary use cases include enhanced recommendations, semantic search, and other knowledge-based applications.

## Features

- **Structural Context Integration**: Embeds graph structure into the Seq2Seq process.
- **Flexible Dataset Support**: Supports multiple datasets (e.g., FB15k-237, Wikidata5M, WN18RR).
- **Efficient Training**: Optimized for batch processing and scalable to large datasets.

## Requirements

- Python 3.7+
- Required packages are specified in `requirements.txt`. Key dependencies include:
  - PyTorch
  - NumPy
  - Pandas
  - transformers==4.40.1

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/hiendang7613/Struct-KGS2S.git
cd Struct-KGS2S
pip install -r requirements.txt
```

## Datasets

The project includes preconfigured support for the following Knowledge Graph datasets:
- **FB15k-237**
- **Wikidata5M**
- **WN18RR**

Each dataset is accessible under its respective license. Please ensure appropriate use.
