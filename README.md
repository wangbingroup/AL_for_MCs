# Cross-system active learning for Li–air catalyst discovery
This repository contains the code accompanying the manuscript:
**Cross-system active learning guided by Li-O2 and Li-CO2 models for discovery of catalysts for Li-air batteries**

## Overview
This work combines information from Li–CO₂ and Li–O₂ batteries with active learning and experimental feedback to guide the discovery of cathode catalysts for Li–air batteries.

The computational workflow consists of three modules:
1. Prediction of structural descriptors.
2. Prediction of Li–CO₂ and Li–O₂ overpotentials.
3. Prediction of Li–air overpotentials using the outputs of the two source-system models.

## Repository structure
| Directory | Description |
| --- | --- |
| `structure/` | Phase classification and prediction of structural descriptors. |
| `CO2_O2/` | Training, evaluation, and prediction of Li–CO₂ and Li–O₂ overpotential models. |
| `air/` | Training, evaluation, uncertainty estimation, and candidate screening for the Li–air model. |

## Requirements
The code is implemented in Python using Jupyter notebooks.

Main packages used across the workflow include:

- NumPy
- pandas
- scikit-learn
- PyTorch
- TabPFN
- NGBoost
- Matplotlib


## Data availability

The datasets supporting the findings of this study are available from the corresponding author upon reasonable request.