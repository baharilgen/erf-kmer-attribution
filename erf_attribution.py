"""
ERF-aware k-mer attribution for DiCNN-UniK.

Extracts convolutional activation windows and aggregates class-level
attributions using activation strength and sample frequency. Also
supports corpus-level k-mer frequency analysis and attribution
stability testing under simulated nucleotide ambiguity.

Usage:
    python erf_attribution.py \
        --model_path DiCNN_Kmer_Classifier_artifacts/DiCNN_Kmer_Classifier_weights.h5 \
        --tokenizer_dir DiCNN_Kmer_Classifier_artifacts/ \
        --label_classes DiCNN_Kmer_Classifier_artifacts/label_classes.json \
        --data_path Flavi_training_data.csv \
        --output_dir attribution_results/ \
        --confidence 0.9 \
        --stability
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)



# Helper functions


def generate_kmers(sequence: str, k: int) -> list:
    if len(sequence) < k:
        return []
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def load_tokenizers(tokenizer_dir: str, k_sizes: list) -> dict:
    tokenizers = {}
    for k in k_sizes:
        path = os.path.join(tokenizer_dir, f"tokenizer_k{k}.json")
        with open(path, encoding="utf-8") as f:
            tokenizers[k] = tokenizer_from_json(json.loads(f.read()))
    print(f"**Tokenizers loaded for k = {k_sizes}")
    return tokenizers


def load_label_classes(path: str) -> list:
    with open(path) as f:
        classes = json.load(f)
    print(f"**Classes: {classes}")
    return classes



# Model input lengths

def get_max_lens_from_model(model, k_sizes: list) -> dict:
    """
    Read input sequence lengths from the model's input layers.

    Supports different Keras versions by checking batch_shape
    and falling back to shape when necessary.
    """
    max_lens = {}
    for k in k_sizes:
        layer_name = f"input_k{k}"
        try:
            # Keras 2 / TF2 compatible
            max_len = int(model.get_layer(layer_name).batch_shape[1])
        except AttributeError:
            # Fallback for newer Keras versions
            max_len = int(model.get_layer(layer_name).shape[1])
        max_lens[k] = max_len
        print(f"  input_k{k} → max_len = {max_len}")
    return max_lens



# Sequence encoding


def encode_sequences(sequences: list, tokenizers: dict,
                     k_sizes: list, max_lens: dict) -> dict:
    encoded = {}
    for k in k_sizes:
        kmers  = [generate_kmers(seq, k) for seq in sequences]
        seqs   = tokenizers[k].texts_to_sequences(kmers)
        padded = pad_sequences(seqs, maxlen=max_lens[k], padding="post")
        encoded[f"input_k{k}"] = padded
    return encoded


# Confidence filtering


def get_correct_high_confidence_mask(model, encoded_inputs: dict,
                                      true_labels: list,
                                      label_classes: list,
                                      confidence_threshold: float = 0.9,
                                      batch_size: int = 64) -> np.ndarray:
    """
    Returns boolean mask: True where model prediction is correct AND
    softmax confidence >= threshold.
    Only these samples are used for attribution aggregation.
    """
    input_list = [encoded_inputs[f"input_k{k}"] for k in sorted(
                  [int(k.split("_k")[1]) for k in encoded_inputs])]

    probs = model.predict(input_list, batch_size=batch_size, verbose=0)
    pred_indices = np.argmax(probs, axis=1)
    pred_confidence = probs.max(axis=1)

    try:
        true_indices = np.array([label_classes.index(l) for l in true_labels])
    except ValueError as e:
        raise ValueError(
            f"Label mismatch between data and label_classes.json: {e}\n"
            f"Data labels found:  {sorted(set(true_labels))}\n"
            f"label_classes.json: {label_classes}"
        )

    correct = pred_indices == true_indices
    confident = pred_confidence >= confidence_threshold
    mask = correct & confident

    print(f"  Correct predictions:          {correct.sum()} / {len(correct)}")
    print(f"  High-confidence (≥{confidence_threshold}): {confident.sum()} / {len(confident)}")
    print(f"  Correct AND high-confidence:  {mask.sum()} / {len(mask)}")
    return mask


# Attribution submodels


def build_attribution_models(full_model, k_sizes: list) -> dict:
    """
    Sub-model per (k, kernel) branch outputting Conv1D activations
    BEFORE GlobalMaxPooling.
    Layer names match model_architecture.py conventions.
    """
    conv_layer_names = {
        (5, 3): "conv1d",
        (5, 5): "conv1d_1",
        (6, 3): "conv1d_2",
        (6, 5): "conv1d_3",
    }
    attribution_models = {}
    for k in k_sizes:
        for kernel in [3, 5]:
            layer_name = conv_layer_names[(k, kernel)]
            try:
                conv_layer = full_model.get_layer(layer_name)
                sub = Model(inputs=full_model.inputs,
                            outputs=conv_layer.output,
                            name=f"attr_k{k}_ker{kernel}")
                attribution_models[(k, kernel)] = sub
                print(f"**Sub-model k={k} kernel={kernel} → '{layer_name}'")
            except ValueError:
                print(f"  !!  Layer '{layer_name}' not found, skipping.")
    return attribution_models



# Filter-level attribution


def get_filter_level_attributions(encoded_inputs: dict,
                                   attribution_models: dict,
                                   tokenizers: dict,
                                   k_sizes: list,
                                   sequences: list,
                                   mask: np.ndarray) -> list:
    """
    Extract filter-level attributions for correctly classified,
    high-confidence samples.

    For each CNN branch, identify the position of maximum activation
    for every convolutional filter and map it to the corresponding
    anchor k-mer and full ERF window.

    Retain the maximum activation value for each filter.
    """
    reverse_index = {}
    for k in k_sizes:
        reverse_index[k] = {v: u for u, v in
                            tokenizers[k].word_index.items()}

    results = []
    selected_indices = np.where(mask)[0]

    for i, sample_idx in enumerate(selected_indices):
        if i % 100 == 0:
            print(f"    Attribution: {i}/{len(selected_indices)}", end="\r")

        sample_input = {key: arr[[sample_idx]]
                        for key, arr in encoded_inputs.items()}

        sample_result = {
            "sample_idx": int(sample_idx),
            "attributions": {}
        }

        for (k, kernel), sub_model in attribution_models.items():
            # Convolutional activations: (L_conv, n_filters)
            conv_acts = sub_model.predict(sample_input, verbose=0)[0]

            n_filters = conv_acts.shape[1]

            # Maximum activation and its position for each filter
            filter_best_pos = np.argmax(conv_acts, axis=0)
            filter_best_val = np.max(conv_acts, axis=0)

            filter_attributions = []
            seq_len = encoded_inputs[f"input_k{k}"].shape[1]
            for filt_idx in range(n_filters):
                pos = int(filter_best_pos[filt_idx])
                val = float(filter_best_val[filt_idx])

                # Recover the full ERF window from adjacent k-mer tokens.
                window_end = min(pos + kernel, seq_len)
                window_ids = encoded_inputs[f"input_k{k}"][sample_idx][pos:window_end]
                window_kmers = [
                    reverse_index[k].get(int(t), "<PAD>")
                    for t in window_ids
                    if reverse_index[k].get(int(t), "<PAD>") != "<PAD>"
                ]
                # Primary kmer = first in window (anchor position)
                # Also store full window for ERF analysis
                kmer_str = window_kmers[0] if window_kmers else "<PAD>"

                filter_attributions.append({
                    "filter":         filt_idx,
                    "position":       pos,
                    "kmer":           kmer_str,          # anchor k-mer
                    "window_kmers":   window_kmers,      # full ERF window
                    "erf_nt":         k + (kernel - 1),  # effective receptive field
                    "activation":     val,
                })

            sample_result["attributions"][(k, kernel)] = filter_attributions

        results.append(sample_result)

    print(f"    Attribution: {len(selected_indices)}/{len(selected_indices)} **")
    return results



# Class-level aggregation


def aggregate_class_attributions(attribution_results: list,
                                  labels_filtered: list,
                                  k_sizes: list) -> dict:
    """
    Aggregate positive filter activations by class, CNN branch,
    and attributed motif.

    Mean activation is calculated across filter-level occurrences.
    Sample frequency is the proportion of samples within a class
    in which the motif appears at least once.

    Weighted activation is the product of mean activation and
    sample frequency.
    """
    # class → branch → agg_type → kmer → activation values
    raw_vals = defaultdict(lambda: defaultdict(
        lambda: {"anchor": defaultdict(list), "window": defaultdict(list)}
    ))

    # class → branch → agg_type → kmer → set(sample_idx)
    raw_samples = defaultdict(lambda: defaultdict(
        lambda: {"anchor": defaultdict(set), "window": defaultdict(set)}
    ))

    for sample, label in zip(attribution_results, labels_filtered):
        sample_idx = sample["sample_idx"]

        for (k, kernel), filter_list in sample["attributions"].items():
            # Count each anchor/window only once per sample for prevalence,
            # but keep all positive filter activations for mean_activation.
            for entry in filter_list:
                val = entry["activation"]
                if val <= 0:
                    continue

                anchor = entry["kmer"]

                window_kmers = [
                    km for km in entry.get("window_kmers", [anchor])
                    if km != "<PAD>"
                ]
                window_key = "|".join(window_kmers) if window_kmers else anchor

                if anchor != "<PAD>":
                    raw_vals[label][(k, kernel)]["anchor"][anchor].append(val)
                    raw_samples[label][(k, kernel)]["anchor"][anchor].add(sample_idx)

                if window_key and window_key != "<PAD>":
                    raw_vals[label][(k, kernel)]["window"][window_key].append(val)
                    raw_samples[label][(k, kernel)]["window"][window_key].add(sample_idx)

    scores = defaultdict(lambda: defaultdict(
        lambda: {"anchor": defaultdict(dict), "window": defaultdict(dict)}
    ))

    class_sample_counts = defaultdict(int)
    for label in set(labels_filtered):
        class_sample_counts[label] = labels_filtered.count(label)

    MIN_OCCURRENCE = 5      # filter-level support
    MIN_SAMPLES = 2         # sample-level support; avoids single-sample artefacts

    for label, branch_dict in raw_vals.items():
        n_samples_cls = class_sample_counts[label]

        for branch, type_dict in branch_dict.items():
            for agg_type, kmer_dict in type_dict.items():
                for kmer, vals in kmer_dict.items():
                    occurrence = len(vals)
                    sample_count = len(raw_samples[label][branch][agg_type][kmer])

                    if occurrence < MIN_OCCURRENCE or sample_count < MIN_SAMPLES:
                        continue

                    mean_act = float(np.mean(vals))
                    sample_freq = sample_count / n_samples_cls if n_samples_cls else 0.0
                    weighted_activation = mean_act * sample_freq

                    scores[label][branch][agg_type][kmer] = {
                        "mean_activation":      mean_act,
                        "occurrence_count":     occurrence,
                        "sample_count":         sample_count,
                        "sample_frequency":     sample_freq,
                        "weighted_activation":  weighted_activation,
                    }

    print("  Sample counts per class (after confidence filter):")
    for cls, cnt in sorted(class_sample_counts.items()):
        print(f"    {cls}: {cnt} samples")

    return scores


def get_top_kmers_per_class(class_kmer_scores: dict,
                             top_n: int = 20,
                             agg_type: str = "anchor") -> dict:
    """
    Return the top-N attributed motifs for each class and CNN branch, 
    ranked by frequency-weighted activation.

    Supports anchor k-mers and full ERF-window signatures.
    Only motifs meeting the minimum occurrence and sample
    support thresholds are included.
    """
    summary = {}
    for cls, branch_dict in class_kmer_scores.items():
        summary[cls] = {}
        for branch, type_dict in branch_dict.items():
            kmer_dict = type_dict.get(agg_type, {})
            sorted_kmers = sorted(
                kmer_dict.items(),
                key=lambda x: x[1].get("weighted_activation", x[1]["mean_activation"]),
                reverse=True
            )[:top_n]
            # Return as (kmer, mean_activation) tuples for compatibility
            summary[cls][branch] = [
                (km, stats["mean_activation"]) for km, stats in sorted_kmers
            ]
    return summary


def compute_discriminative_scores(class_kmer_scores: dict,
                                   top_n: int = 20,
                                   epsilon: float = 1e-8) -> pd.DataFrame:
    """
Compute frequency-aware discriminative scores for each class,
CNN branch, and attributed motif.

The score combines mean filter activation with sample-level
frequency and compares the resulting weighted activation
against its mean across the other classes.

    weighted_activation = mean_activation * sample_frequency

    disc_score_log2 = log2(
        (weighted_activation_class + eps) /
        (mean_weighted_activation_other_classes + eps)
    )
"""

    all_classes = list(class_kmer_scores.keys())
    records = []

    for cls in all_classes:
        other_classes = [c for c in all_classes if c != cls]

        for branch, type_dict in class_kmer_scores[cls].items():
            for agg_type in ["anchor", "window"]:
                kmer_dict = type_dict.get(agg_type, {})

                class_records = []
                for kmer, stats_cls in kmer_dict.items():
                    mean_act = stats_cls["mean_activation"]
                    occ_cls = stats_cls["occurrence_count"]
                    sample_count = stats_cls.get("sample_count", np.nan)
                    freq_cls = stats_cls["sample_frequency"]
                    weighted_cls = stats_cls.get(
                        "weighted_activation",
                        mean_act * freq_cls
                    )

                    other_mean_acts = []
                    other_weighted = []
                    other_freqs = []

                    for oc in other_classes:
                        other_stats = (
                            class_kmer_scores[oc]
                            .get(branch, {})
                            .get(agg_type, {})
                            .get(kmer, None)
                        )
                        if other_stats is None:
                            other_mean_acts.append(0.0)
                            other_weighted.append(0.0)
                            other_freqs.append(0.0)
                        else:
                            other_act = other_stats.get("mean_activation", 0.0)
                            other_freq = other_stats.get("sample_frequency", 0.0)
                            other_mean_acts.append(other_act)
                            other_freqs.append(other_freq)
                            other_weighted.append(
                                other_stats.get("weighted_activation",
                                                other_act * other_freq)
                            )

                    mean_other_act = float(np.mean(other_mean_acts)) if other_mean_acts else 0.0
                    mean_other_weighted = float(np.mean(other_weighted)) if other_weighted else 0.0
                    max_other_freq = float(np.max(other_freqs)) if other_freqs else 0.0

                    # Kept for diagnosis only: activation-only score.
                    disc_activation_log2 = float(np.log2(
                        (mean_act + epsilon) / (mean_other_act + epsilon)
                    ))

                    # Frequency-weighted discriminative score.
                    disc_weighted_log2 = float(np.log2(
                        (weighted_cls + epsilon) /
                        (mean_other_weighted + epsilon)
                    ))

                    class_records.append({
                        "class":                    cls,
                        "agg_type":                 agg_type,
                        "k":                        branch[0],
                        "kernel":                   branch[1],
                        "kmer":                     kmer,
                        "mean_activation":          mean_act,
                        "occurrence_count":         occ_cls,
                        "sample_count":             sample_count,
                        "sample_frequency":         round(freq_cls, 4),
                        "weighted_activation":      round(weighted_cls, 6),
                        "activation_other":         mean_other_act,
                        "weighted_activation_other": round(mean_other_weighted, 6),
                        "max_other_frequency":      round(max_other_freq, 4),
                        "disc_activation_log2":     round(disc_activation_log2, 4),
                        "disc_score_log2":          round(disc_weighted_log2, 4),
                    })

                # Retain the highest-scoring motifs.
                class_records = sorted(
                    class_records,
                    key=lambda r: r["disc_score_log2"],
                    reverse=True
                )[:top_n]
                records.extend(class_records)

    df = pd.DataFrame(records)
    df = df.sort_values(["class", "k", "kernel", "disc_score_log2"],
                        ascending=[True, True, True, False])
    return df



# Corpus-level hapax analysis


def compute_corpus_hapax(sequences: list, k_sizes: list) -> dict:
    """
    Compute corpus-level hapax counts.
    Do NOT rely on tokenizer.word_counts - tokenizer may use universal
    library rather than corpus-derived frequencies.
    Returns dict: k → set of hapax k-mer strings (count == 1 in corpus).
    """
    from collections import Counter
    hapax = {}
    for k in k_sizes:
        counter = Counter()
        for seq in sequences:
            for kmer in generate_kmers(seq, k):
                counter[kmer] += 1
        hapax[k] = {km for km, cnt in counter.items() if cnt == 1}
        print(f"  Corpus hapax k={k}: {len(hapax[k])} hapax "
              f"out of {len(counter)} unique k-mers "
              f"(from {len(sequences)} sequences)")
    return hapax


def hapax_attribution_analysis(top_kmers_summary: dict,
                                tokenizers: dict,
                                k_sizes: list,
                                output_dir: str,
                                corpus_hapax: dict = None) -> pd.DataFrame:
    """
    Calculate the fraction of top-attributed k-mers that occur
    exactly once in the training corpus (hapax legomena).

    Uses corpus-level counts when available, otherwise
    falls back to tokenizer word counts.
    """
    records = []
    for k in k_sizes:
        # Prefer corpus-derived hapax over tokenizer counts
        if corpus_hapax and k in corpus_hapax:
            hapax_set = corpus_hapax[k]
            print(f"  k={k}: using corpus hapax ({len(hapax_set)} hapax k-mers)")
        else:
            tok = tokenizers[k]
            if not hasattr(tok, "word_counts"):
                print(f"  ⚠️  tokenizer k={k} has no word_counts — skipping hapax.")
                continue
            hapax_set = {w for w, c in tok.word_counts.items() if c == 1}
            print(f"  k={k}: {len(hapax_set)} hapax k-mers "
                  f"out of {len(tok.word_counts)} total")

        for cls, branch_dict in top_kmers_summary.items():
            for (bk, kernel), entries in branch_dict.items():
                if bk != k:
                    continue
                top_kmers_list = [e[0] for e in entries]
                n_hapax = sum(1 for km in top_kmers_list if km in hapax_set)
                frac    = n_hapax / len(top_kmers_list) if top_kmers_list else 0.0
                records.append({
                    "class":       cls,
                    "k":           k,
                    "kernel":      kernel,
                    "n_top":       len(top_kmers_list),
                    "n_hapax":     n_hapax,
                    "frac_hapax":  frac,
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df.to_csv(os.path.join(output_dir, "hapax_attribution.csv"), index=False)

    # Plot
    pivot = df.groupby("class")["frac_hapax"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pivot["class"], pivot["frac_hapax"], color="coral", edgecolor="k")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="50% line")
    ax.set_xlabel("Flavivirus Class")
    ax.set_ylabel("Fraction of Top k-mers that are Hapax Legomena")
    ax.set_title("Hapax Legomenon Contribution to Attribution per Class\n"
                 "(hapax = k-mer appearing exactly once in training corpus)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hapax_attribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  📊 Saved: hapax_attribution.png")
    return df



	# Attribution stability under nucleotide ambiguity


def simulate_ambiguity(sequence: str, ambiguity_rate: float,
                       seed: int = 42) -> str:
    """
    Replace `ambiguity_rate` fraction of nucleotides with 'N'.
    This simulates nucleotide ambiguity, NOT real low genomic coverage.
    Label accordingly in paper: 'simulated nucleotide ambiguity'.
    """
    rng  = np.random.default_rng(seed)
    chars = list(sequence)
    n_mask = int(len(chars) * ambiguity_rate)
    positions = rng.choice(len(chars), size=n_mask, replace=False)
    for p in positions:
        chars[p] = "N"
    return "".join(chars)


def ambiguity_stability_analysis(sequences: list,
                                  labels: list,
                                  tokenizers: dict,
                                  attribution_models: dict,
                                  k_sizes: list,
                                  max_lens: dict,
                                  mask: np.ndarray = None,
                                  ambiguity_rates: list = None,
                                  n_samples: int = 500,
                                  per_class: int = 50) -> pd.DataFrame:
    """
    Jaccard similarity of top-10 attributed k-mers between
    clean sequence and sequence with simulated ambiguity.
    """
    if ambiguity_rates is None:
        # ambiguity_rate = 1 - coverage (e.g. 0.3 → 70% coverage)
        ambiguity_rates = [0.0, 0.3, 0.5, 0.7, 0.8]

    # Stratified sampling: sample equally from each class
    # so rare classes (e.g. TBE, Yellow Fever) are not underrepresented
    valid_indices = np.where(mask)[0] if mask is not None else np.arange(len(sequences))
    rng = np.random.default_rng(0)

    # Group valid indices by class
    from collections import defaultdict as _dd
    class_indices = _dd(list)
    for i in valid_indices:
        class_indices[labels[i]].append(i)

    # Per-class sample size
    n_classes = len(class_indices)
    per_class = min(per_class, max(1, n_samples // n_classes))

    idx = []
    for cls, cls_idx in class_indices.items():
        cls_arr = np.array(cls_idx)
        n_take = min(per_class, len(cls_arr))
        chosen = rng.choice(cls_arr, size=n_take, replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    print(f"  Stratified stability sampling: {len(idx)} samples "
          f"({per_class} per class, {n_classes} classes)")

    reverse_index = {}
    for k in k_sizes:
        reverse_index[k] = {v: u for u, v in
                            tokenizers[k].word_index.items()}

    def top_kmers_for_sample(seq, k, kernel, use_anchor=True):
        """
        Extract top-10 filter attributions from a sequence.
        use_anchor=True  → returns anchor k-mers only (less brittle, main metric)
        use_anchor=False → returns full ERF-window strings (exact, supplementary)
        """
        enc  = encode_sequences([seq], tokenizers, k_sizes, max_lens)
        acts = attribution_models[(k, kernel)].predict(
            enc, verbose=0)[0]                        # (L_conv, n_filters)
        best_pos = np.argmax(acts, axis=0)            # (n_filters,)
        best_val = np.max(acts, axis=0)               # (n_filters,)
        top10_filters = np.argsort(best_val)[::-1][:10]
        kmers = set()
        seq_len = enc[f"input_k{k}"].shape[1]
        for f in top10_filters:
            pos = int(best_pos[f])
            if use_anchor:
                # Anchor k-mer only — less brittle under ambiguity
                token_id = int(enc[f"input_k{k}"][0][pos])
                anchor = reverse_index[k].get(token_id, "<PAD>")
                if anchor != "<PAD>":
                    kmers.add(anchor)
            else:
                # Full ERF-window — exact, used for supplementary
                window_end = min(pos + kernel, seq_len)
                window_ids = enc[f"input_k{k}"][0][pos:window_end]
                window_kmers = [
                    reverse_index[k].get(int(t), "<PAD>")
                    for t in window_ids
                ]
                window_kmers = [km for km in window_kmers if km != "<PAD>"]
                window_key = "|".join(window_kmers)
                if window_key:
                    kmers.add(window_key)
        return kmers

    records = []
    for i, sample_idx in enumerate(idx):
        seq   = sequences[sample_idx]
        label = labels[sample_idx]
        print(f"  Stability: {i+1}/{len(idx)}", end="\r")

        for k in k_sizes:
            for kernel in [3, 5]:
                if (k, kernel) not in attribution_models:
                    continue
                ref_top = top_kmers_for_sample(seq, k, kernel)

                for rate in ambiguity_rates:
                    masked_seq = simulate_ambiguity(seq, rate,
                                                    seed=int(sample_idx))

                    # Anchor-level Jaccard (main metric — less brittle)
                    ref_anchor  = top_kmers_for_sample(seq, k, kernel, use_anchor=True)
                    mask_anchor = top_kmers_for_sample(masked_seq, k, kernel, use_anchor=True)
                    union_a = ref_anchor | mask_anchor
                    inter_a = ref_anchor & mask_anchor
                    jaccard_anchor = len(inter_a) / len(union_a) if union_a else 1.0

                    # Window-level Jaccard (supplementary — exact, brittle)
                    ref_window  = top_kmers_for_sample(seq, k, kernel, use_anchor=False)
                    mask_window = top_kmers_for_sample(masked_seq, k, kernel, use_anchor=False)
                    union_w = ref_window | mask_window
                    inter_w = ref_window & mask_window
                    jaccard_window = len(inter_w) / len(union_w) if union_w else 1.0

                    records.append({
                        "sample_idx":        int(sample_idx),
                        "label":             label,
                        "ambiguity_rate":    rate,
                        "retained_fraction": 1.0 - rate,
                        "k":                 k,
                        "kernel":            kernel,
                        "jaccard_anchor":    jaccard_anchor,   # main metric
                        "jaccard_window":    jaccard_window,   # supplementary
                    })

    print(f"  Stability: {len(idx)}/{len(idx)} **")
    return pd.DataFrame(records)


def plot_stability(stability_df: pd.DataFrame, output_dir: str) -> None:
    branches = stability_df[["k", "kernel"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(branches),
                             figsize=(5 * len(branches), 4), sharey=True)
    if len(branches) == 1:
        axes = [axes]

    for ax, (k, kernel) in zip(axes, branches):
        sub  = stability_df[(stability_df["k"] == k) &
                            (stability_df["kernel"] == kernel)]
        # Anchor-level (main) — less brittle
        grp_a  = sub.groupby("retained_fraction")["jaccard_anchor"]
        mean_a = grp_a.mean()
        std_a  = grp_a.std()

        # Window-level (supplementary) — exact match
        grp_w  = sub.groupby("retained_fraction")["jaccard_window"]
        mean_w = grp_w.mean()

        ax.plot(mean_a.index, mean_a.values, marker="o",
                color="steelblue", label="Anchor-level (main)")
        ax.fill_between(mean_a.index,
                        (mean_a - std_a).clip(0, 1),
                        (mean_a + std_a).clip(0, 1),
                        alpha=0.2, color="steelblue")
        ax.plot(mean_w.index, mean_w.values, marker="s",
                color="coral", ls="--", lw=1.2, label="ERF-window (supplementary)")
        ax.set_xlabel("Retained Nucleotide Fraction\n(under simulated ambiguity)")
        ax.set_ylabel("Jaccard Similarity")
        ax.set_title(f"Attribution Stability\nk={k}, kernel={kernel}")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, ls="--", color="gray", lw=0.8)
        ax.legend(fontsize=7)

    plt.suptitle("ERF-window Attribution Stability Under Simulated Nucleotide Ambiguity",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(output_dir, "coverage_stability.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# Visualisation

def plot_top_kmers(top_kmers_summary: dict, output_dir: str,
                   branch: tuple, top_n: int = 15) -> None:
    """
    Split into two figures:
      Fig A — Dengue serotypes (4 panels)
      Fig B — Other flaviviruses (6 panels)
    ERF-window keys shown for window agg_type; anchor k-mers for anchor.
    Common x-axis limit per figure for fair visual comparison.
    """
    DENGUE  = ["Dengue_1", "Dengue_2", "Dengue_3", "Dengue_4"]
    OTHERS  = ["Japanese Encephalitis", "Tick-Borne Encephalitis",
               "Usutu Virus", "West Nile Virus", "Yellow Fever", "Zika Virus"]

    def _make_figure(class_list, fig_label, suffix):
        n_cols = min(2, len(class_list))
        n_rows = (len(class_list) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()

        # Compute global x-max for common scale
        global_max = 0
        for cls in class_list:
            entries = top_kmers_summary.get(cls, {}).get(branch, [])[:top_n]
            if entries:
                global_max = max(global_max, max(e[1] for e in entries))
        x_lim = global_max * 1.1 if global_max > 0 else 1.0

        for ax, cls in zip(axes, class_list):
            entries = top_kmers_summary.get(cls, {}).get(branch, [])[:top_n]
            if not entries:
                ax.set_visible(False)
                continue

            labels = [e[0] for e in entries]
            scores = [e[1] for e in entries]

            # Color ERF-window keys differently to highlight contribution
            colors = []
            for lbl in labels:
                if "|" in lbl:
                    colors.append("darkorange")   # ERF-window
                else:
                    colors.append("steelblue")    # anchor k-mer

            bars = ax.barh(labels[::-1], scores[::-1],
                           color=colors[::-1], edgecolor="k", linewidth=0.4)
            ax.set_title(f"{cls}\n(k={branch[0]}, kernel={branch[1]})",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean Activation", fontsize=10)
            ax.tick_params(axis="y", labelsize=9)
            ax.set_xlim(0, x_lim)
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axes[len(class_list):]:
            ax.set_visible(False)

        # Legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="steelblue",  edgecolor="k", label="Anchor k-mer"),
            Patch(facecolor="darkorange", edgecolor="k", label="ERF-window signature"),
        ]
        fig.legend(handles=legend_elements, loc="upper right",
                   fontsize=9, frameon=True)

        plt.suptitle(
            f"Top activation-associated k-mers — {fig_label}\n"
            f"(k={branch[0]}, kernel={branch[1]}, "
            f"min 5 filter-level occurrences, correct & high-confidence only)",
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        out = os.path.join(output_dir,
                           f"top_kmers_k{branch[0]}_ker{branch[1]}_{suffix}.png")
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print(f" Saved: {out}")

    _make_figure(DENGUE,  "Dengue Serotypes",   "dengue")
    _make_figure(OTHERS,  "Other Flaviviruses",  "other")


def plot_heatmap(top_kmers_summary: dict, output_dir: str,
                 branch: tuple, top_n: int = 15) -> None:
    """
    Heatmap of top-N activation-associated k-mers per class.
    - top_n=15 for readability
    - Sorted rows by total activation
    - Larger fonts
    - Column-wise normalization noted in title
    - Grid lines for readability
    """
    all_kmers = set()
    for cls in top_kmers_summary:
        for kmer, _ in top_kmers_summary[cls].get(branch, [])[:top_n]:
            all_kmers.add(kmer)

    classes   = sorted(top_kmers_summary.keys())
    kmer_list = list(all_kmers)
    matrix    = np.zeros((len(kmer_list), len(classes)))

    for j, cls in enumerate(classes):
        score_dict = dict(top_kmers_summary[cls].get(branch, []))
        for i, km in enumerate(kmer_list):
            matrix[i, j] = score_dict.get(km, 0.0)

    # Sort rows by total activation — most active on top
    row_order  = np.argsort(matrix.sum(axis=1))[::-1]
    matrix     = matrix[row_order]
    kmer_list  = [kmer_list[i] for i in row_order]

    # Column-wise normalization within class
    col_max = matrix.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1
    matrix_norm = matrix / col_max

    fig_w = max(10, len(classes) * 1.4)
    fig_h = max(6, len(kmer_list) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalised Activation\n(column-wise, within class)", fontsize=10)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=11)
    ax.set_yticks(range(len(kmer_list)))
    ax.set_yticklabels(kmer_list, fontsize=10)

    # Minor grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(kmer_list), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)

    ax.set_title(
        f"Activation-associated k-mer Heatmap (k={branch[0]}, kernel={branch[1]})\n"
        f"Column-wise normalised | correct & high-confidence samples only",
        fontsize=12, pad=12
    )
    plt.tight_layout()
    out = os.path.join(output_dir,
                       f"kmer_heatmap_k{branch[0]}_ker{branch[1]}.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")


def plot_erf_window_table(disc_df: pd.DataFrame,
                           output_dir: str,
                           branch_k: int = 5,
                           branch_kernel: int = 3,
                           top_n: int = 3) -> None:
    """
    Table figure showing top ERF-window signatures per class.
    Makes ERF-window attribution contribution explicit for reviewer.
    Only window agg_type entries containing | separator are shown.
    """
    sub = disc_df[
        (disc_df["agg_type"] == "window") &
        (disc_df["k"] == branch_k) &
        (disc_df["kernel"] == branch_kernel) &
        (disc_df["kmer"].str.contains("|", regex=False))
    ].copy()

    if sub.empty:
        print("  ! No ERF-window entries found for table plot.")
        return

    rows = []
    for cls, grp in sub.groupby("class"):
        top = grp.nlargest(top_n, "disc_score_log2")
        for _, row in top.iterrows():
            rows.append({
                "Class":        cls,
                "ERF-window":   row["kmer"],
                "Mean Act.":    f"{row['mean_activation']:.3f}",
                "log2 Score":   f"{row['disc_score_log2']:.2f}",
                "Sample Freq.": f"{row['sample_frequency']:.2f}",
            })

    if not rows:
        print("  !!  No window rows to plot.")
        return

    df_plot   = pd.DataFrame(rows)
    n_rows    = len(df_plot)
    col_labels = list(df_plot.columns)

    fig, ax = plt.subplots(figsize=(13, max(4, n_rows * 0.42)))
    ax.axis("off")

    table = ax.table(
        cellText=df_plot.values.tolist(),
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    # Header style
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c5f8a")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, n_rows + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    ax.set_title(
        f"Top ERF-window Attribution Signatures per Flavivirus Class\n"
        f"(k={branch_k}, kernel={branch_kernel}, "
        f"ERF={branch_k + branch_kernel - 1} nt, window-level attribution)",
        fontsize=11, pad=10
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "erf_window_table.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {out}")



# Main


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    k_sizes = [5, 6]

    print("\n[1/8] Loading model …")
    model = tf.keras.models.load_model(args.model_path)

    print("\n[2/8] Reading max_lens from model input shapes …")
    max_lens = get_max_lens_from_model(model, k_sizes)

    print("\n[3/8] Loading tokenizers and label classes …")
    tokenizers    = load_tokenizers(args.tokenizer_dir, k_sizes)
    label_classes = load_label_classes(args.label_classes)

    print("\n[4/8] Loading data …")
    df        = pd.read_csv(args.data_path)
    sequences = df["Sequence"].tolist()
    labels    = df["Organism_Name"].tolist()

    if args.n_samples and args.n_samples < len(sequences):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(sequences), args.n_samples, replace=False)
        sequences = [sequences[i] for i in idx]
        labels    = [labels[i]    for i in idx]
        print(f"  Subsampled to {args.n_samples} sequences.")

    print("\n[5/8] Encoding sequences …")
    encoded_inputs = encode_sequences(sequences, tokenizers, k_sizes, max_lens)

    print("\n[6/8] Filtering to correct, high-confidence predictions …")
    mask = get_correct_high_confidence_mask(
        model, encoded_inputs, labels, label_classes,
        confidence_threshold=args.confidence
    )
    labels_filtered = [labels[i] for i in np.where(mask)[0]]

    print("\n[7/8] Building attribution sub-models …")
    attribution_models = build_attribution_models(model, k_sizes)

    print("\n[8/8] Running filter-level attribution …")
    attribution_results = get_filter_level_attributions(
        encoded_inputs, attribution_models, tokenizers,
        k_sizes, sequences, mask
    )

    # Aggregate
    class_kmer_scores = aggregate_class_attributions(
        attribution_results, labels_filtered, k_sizes
    )
    top_kmers_summary = get_top_kmers_per_class(class_kmer_scores, top_n=20)

    # Save raw activation CSV
    rows = []
    for cls, branch_dict in top_kmers_summary.items():
        for (k, kernel), entries in branch_dict.items():
            for rank, (kmer, score) in enumerate(entries, 1):
                rows.append({"class": cls, "k": k, "kernel": kernel,
                             "rank": rank, "kmer": kmer,
                             "mean_activation": score})
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "top_kmer_attributions.csv"),
        index=False
    )
    print("  ** Saved: top_kmer_attributions.csv")

    # Compute and save discriminative scores
    print("\n  Computing discriminative scores …")
    disc_df = compute_discriminative_scores(class_kmer_scores, top_n=20)
    disc_path = os.path.join(args.output_dir, "discriminative_kmer_scores.csv")
    disc_df.to_csv(disc_path, index=False)
    print(f"  ** Saved: discriminative_kmer_scores.csv")
    print("  Top discriminative k-mers per class (k=5, kernel=3):")
    sample = disc_df[(disc_df["k"]==5) & (disc_df["kernel"]==3)]
    for cls, grp in sample.groupby("class"):
        top3 = grp.head(3)[["kmer","disc_score_log2"]].values
        print(f"    {cls}: " + ", ".join(f"{km}({sc:.1f})" for km, sc in top3))

    # Plots
    for branch in [(5, 3), (5, 5), (6, 3), (6, 5)]:
        plot_top_kmers(top_kmers_summary, args.output_dir, branch)
        plot_heatmap(top_kmers_summary, args.output_dir, branch)

    # Hapax — FIX 3: compute from real corpus, not tokenizer
    print("\n  Computing corpus-level hapax legomena ...")
    corpus_hapax = compute_corpus_hapax(sequences, k_sizes)
    hapax_attribution_analysis(top_kmers_summary, tokenizers,
                                k_sizes, args.output_dir,
                                corpus_hapax=corpus_hapax)

    # Stability (optional)
    if args.stability:
        print("\n  Running ambiguity stability analysis …")
        stab_df = ambiguity_stability_analysis(
            sequences, labels, tokenizers, attribution_models,
            k_sizes, max_lens,
            mask=mask,
            ambiguity_rates=[0.0, 0.3, 0.5, 0.7, 0.8],
            n_samples=min(500, len(sequences)),
            per_class=args.per_class
        )
        stab_df.to_csv(
            os.path.join(args.output_dir, "coverage_stability.csv"),
            index=False
        )
        plot_stability(stab_df, args.output_dir)

    print(f"\n** Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ERF-aware k-mer attribution for DiCNN-UniK"
    )
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--label_classes", required=True)
    parser.add_argument("--data_path",     required=True)
    parser.add_argument("--output_dir",    default="attribution_results/")
    parser.add_argument("--confidence",    type=float, default=0.9,
                        help="Min softmax confidence for aggregation (default 0.9)")
    parser.add_argument("--n_samples",     type=int,   default=None)
    parser.add_argument("--stability",     action="store_true")
    parser.add_argument("--per_class",     type=int,   default=50,
                        help="Samples per class for stability analysis (default: 50)")
    args = parser.parse_args()
    main(args)
