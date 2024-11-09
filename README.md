

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

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Struct-KGS2S.git
cd Struct-KGS2S
pip install -r requirements.txt
```

## Datasets

The project includes preconfigured support for the following Knowledge Graph datasets:
- **FB15k-237**
- **Wikidata5M**
- **WN18RR**

Each dataset is accessible under its respective license. Please ensure appropriate use.

## Training & Evaluation

Train and evaluate Struct-KGS2S using the provided notebooks:

1. **FB15k-237**: `training_evaluate_fb15k237.ipynb`
2. **Wikidata5M**: `training_evaluate_wikidata5M.ipynb`
3. **WN18RR**: `training_evaluate_wn18rr.ipynb`

These notebooks guide you through setting up, training, and evaluating the model for each dataset. 

## Usage

1. Prepare the dataset as per instructions.
2. Run the notebooks to train and evaluate the model.
3. Adjust hyperparameters as needed to optimize performance on your specific dataset.

## Results

Struct-KGS2S demonstrates promising improvements in link prediction tasks across multiple benchmark datasets. Detailed results can be found in each notebook's output section.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We extend our gratitude to contributors and those who provided datasets. This project builds upon prior work in link prediction and Knowledge Graph completion.

