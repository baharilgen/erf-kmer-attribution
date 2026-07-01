# ERF-aware k-mer Attribution Framework

Architecture-aware explainability framework for convolutional neural networks operating on genomic k-mer representations.

---

## Overview

This repository contains the implementation of an ERF-aware attribution framework for interpreting genomic CNN classifiers.

The framework extends the previously published **DiCNN-UniK** architecture with an interpretable attribution pipeline that traces highly activated convolutional filters back to their **Effective Receptive Fields (ERFs)**, providing biologically interpretable sequence-level explanations.

Unlike conventional feature-level attribution methods, the proposed framework explicitly exploits the receptive field structure of convolutional neural networks to identify discriminative genomic sequence signatures.

---

## Features

- ERF-aware activation attribution
- Class-level attribution aggregation
- Frequency-aware discriminative scoring
- Attribution robustness analysis under simulated nucleotide ambiguity
- Occlusion-based perturbation validation

---

## Repository Structure

The repository is being prepared for public release. Planned components include:

```text
erf-kmer-attribution/
│
├── scripts/
│   ├── activation_analysis.py
│   └── occlusion_validation.py
│
├── config/
│   ├── tokenizer_k5.json
│   ├── tokenizer_k6.json
│   └── label_classes.json
│
├── models/
│   └── DiCNN_Kmer_Classifier_weights.h5
│
├── example_results/
│
├── requirements.txt
└── README.md
```
```markdown
Code, configuration files, model weights, and example outputs will be added progressively as they become available and approved for public release.

---



## Data

The genomic sequences used in this study were obtained from the publicly available **NCBI Virus** database.

---

## Citation

If you use this repository, please cite the accompanying publication:

İlgen B., Shahid A., Kühnert D.
*Interpretable Flavivirus Classification under Low-Coverage Conditions via ERF-Aware k-mer Attribution*.
EXPLIMED 2026.

---

## Acknowledgements

This repository provides the attribution framework developed in this work.

The underlying **DiCNN-UniK** classifier is based on the previously published architecture by Shahid *et al.* and is extended here with architecture-aware explainability and validation modules.
