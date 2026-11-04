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
│   ├── erf_attribution.py
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
## Code

- `erf_attribution.py` — ERF-aware activation attribution, class-level aggregation, discriminative scoring, hapax analysis, and attribution stability testing.
- `occlusion_validation.py` — Occlusion-based perturbation analysis for evaluating attributed sequence regions.

## Model Requirements

The scripts require a trained DiCNN-UniK model, the corresponding k-mer tokenizers, and label mappings. These files are part of the underlying DiCNN-UniK implementation and are not included in this repository.

## Usage

Run the attribution analysis with a trained DiCNN-UniK model and the corresponding tokenizers and label mappings.

```bash
python erf_attribution.py \
    --model_path DiCNN_Kmer_Classifier_artifacts/DiCNN_Kmer_Classifier_weights.h5 \
    --tokenizer_dir DiCNN_Kmer_Classifier_artifacts/ \
    --label_classes DiCNN_Kmer_Classifier_artifacts/label_classes.json \
    --data_path Flavi_training_data.csv \
    --output_dir attribution_results/ \
    --confidence 0.9 \
    --stability
```

The input CSV must contain `Sequence` and `Organism_Name` columns. The optional `--stability` flag enables analysis under simulated nucleotide ambiguity.


## Data

The genomic sequences used in this study were obtained from the publicly available **NCBI Virus** database.

---

## Citation

If you use this repository, please cite the accompanying publication:

İlgen B., Shahid A., Kühnert D.
*Interpretable Flavivirus Classification under Low-Coverage Conditions via ERF-Aware k-mer Attribution*.
 IJCAI-ECAI 2026 Workshop on Explainable Artificial Intelligence for the Medical Domain (EXPLIMED 2026). 2026.

---

## Acknowledgements

This repository provides the attribution framework developed in this work.

The underlying **DiCNN-UniK** classifier is based on the previously published architecture by Shahid *et al.* and is extended here with architecture-aware explainability and validation modules.
