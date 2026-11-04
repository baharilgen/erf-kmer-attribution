# ERF-aware k-mer Attribution for DiCNN-UniK

Code accompanying the paper Interpretable Flavivirus Classification under Low-Coverage Conditions via ERF-Aware k-mer Attribution (EXPLIMED 2026).

---

## Overview

This repository provides an attribution analysis pipeline for DiCNN-UniK, a convolutional neural network for genomic sequence classification using k-mer representations.

The method maps convolutional filter activations to their corresponding k-mers and effective receptive field (ERF) windows. It aggregates these attributions across samples to identify class-associated sequence patterns and quantify their discriminative scores.

---

## Features

- ERF-aware attribution of convolutional filter activations
- Extraction of anchor k-mers and full ERF windows
- Class-level aggregation of activation strength and sample frequency
- Frequency-weighted discriminative scoring
- Corpus-level hapax legomena analysis
- Attribution stability analysis under simulated nucleotide ambiguity
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
