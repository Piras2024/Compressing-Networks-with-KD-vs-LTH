# Compressing Neural Networks: Knowledge Distillation vs Lottery Ticket Hypothesis

This repository contains the implementation and experiments for my project comparing two model compression techniques: **Knowledge Distillation (KD)** and the **Lottery Ticket Hypothesis (LTH)**. The goal is to investigate how different compression strategies affect predictive performance, training efficiency, and learned representations.

## Project Overview

Modern neural networks are often overparameterized, making model compression crucial for efficient deployment. This project compares:

- **Knowledge Distillation (KD):** Training a smaller student network to mimic a larger teacher model.  
- **Lottery Ticket Hypothesis (LTH):** Identifying sparse subnetworks within a dense model that can train effectively on their own.

Experiments are performed on the **MNIST1D dataset**, which provides a non-trivial yet fast-to-train classification task. Multiple students and tickets are trained to analyze variability across compressed models.

Beyond predictive performance and training time, the project analyzes:

- **Representation Similarity:** Using **Centered Kernel Alignment (CKA)** to compare learned feature representations.  
- **Feature Sparsity and Effective Dimensionality:** Measuring how pruning affects the penultimate layer activations.  
- **Linear Mode Connectivity (LMC):** Investigating the geometry of the loss landscape in parameter space.

## Results Summary

| Method | Test Accuracy (%) | Training Time (min) |
|--------|-----------------|------------------|
| KD     | 96.68           | 2                |
| LTH    | 98.89           | 12               |

- **Lottery Tickets** achieve higher accuracy and maintain richer penultimate-layer features, despite heavy parameter sparsity.  
- **Distilled students** are constrained by architectural bottlenecks, resulting in lower feature dimensionality and representation similarity.  
- LMC is satisfied only when using **matching models** with early weight rewinding, highlighting the role of early training dynamics.

## Repository Structure
# Compressing Neural Networks: Knowledge Distillation vs Lottery Ticket Hypothesis

This repository contains the implementation and experiments for my project comparing two model compression techniques: **Knowledge Distillation (KD)** and the **Lottery Ticket Hypothesis (LTH)**. The goal is to investigate how different compression strategies affect predictive performance, training efficiency, and learned representations.

## Project Overview

Modern neural networks are often overparameterized, making model compression crucial for efficient deployment. This project compares:

- **Knowledge Distillation (KD):** Training a smaller student network to mimic a larger teacher model.  
- **Lottery Ticket Hypothesis (LTH):** Identifying sparse subnetworks within a dense model that can train effectively on their own.

Experiments are performed on the **MNIST1D dataset**, which provides a non-trivial yet fast-to-train classification task. Multiple students and tickets are trained to analyze variability across compressed models.

Beyond predictive performance and training time, the project analyzes:

- **Representation Similarity:** Using **Centered Kernel Alignment (CKA)** to compare learned feature representations.  
- **Feature Sparsity and Effective Dimensionality:** Measuring how pruning affects the penultimate layer activations.  
- **Linear Mode Connectivity (LMC):** Investigating the geometry of the loss landscape in parameter space.

## Results Summary

| Method | Test Accuracy (%) | Training Time (min) |
|--------|-----------------|------------------|
| KD     | 96.68           | 2                |
| LTH    | 98.89           | 12               |

- **Lottery Tickets** achieve higher accuracy and maintain richer penultimate-layer features, despite heavy parameter sparsity.  
- **Distilled students** are constrained by architectural bottlenecks, resulting in lower feature dimensionality and representation similarity.  
- LMC is satisfied only when using **matching models** with early weight rewinding, highlighting the role of early training dynamics.
