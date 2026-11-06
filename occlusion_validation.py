"""
occlusion_validation.py

Sequence-level occlusion validation for DiCNN-UniK ERF-window attribution.

  - Masks nucleotides with 'N' and re-encodes both k=5 and k=6 branches.
  - Selects the highest-activation filter-specific ERF positions
    from a configurable attribution branch.
  - Compares attributed ERF masking against a random contiguous-block
    baseline matched by the total number of masked nucleotides.
  - Evaluates perturbation sensitivity using changes in classification
    confidence and logit margins.
  - Supports motif-family occlusion and class-level summaries.

Note:
  The random-block baseline matches the number of masked nucleotides,
  but does not necessarily preserve the spatial distribution of
  the attributed ERF regions.

Usage:
    python occlusion_validation.py \
        --model_path DiCNN_Kmer_Classifier_artifacts/DiCNN_Kmer_Classifier_weights.h5 \
        --tokenizer_dir DiCNN_Kmer_Classifier_artifacts/ \
        --label_classes DiCNN_Kmer_Classifier_artifacts/label_classes.json \
        --data_path Flavi_training_data.csv \
        --output_dir occlusion_results/ \
        --confidence 0.9 \
        --n_samples 500 \
        --n_random 15 \
        --n_top_filters 5 \
        --attribution_k 6 \
        --attribution_kernel 5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
    print(f"** Tokenizers loaded for k = {k_sizes}")
    return tokenizers


def load_label_classes(path: str) -> list:
    with open(path) as f:
        classes = json.load(f)
    print(f"** Classes: {classes}")
    return classes


def get_max_lens_from_model(model, k_sizes: list) -> dict:
    max_lens = {}
    for k in k_sizes:
        try:
            max_len = int(model.get_layer(f"input_k{k}").batch_shape[1])
        except AttributeError:
            max_len = int(model.get_layer(f"input_k{k}").shape[1])
        max_lens[k] = max_len
    return max_lens


def encode_single(sequence: str, tokenizers: dict,
                  k_sizes: list, max_lens: dict) -> dict:
    """Encode a single sequence into model inputs for both branches."""
    encoded = {}
    for k in k_sizes:
        kmers  = generate_kmers(sequence, k)
        seqs   = tokenizers[k].texts_to_sequences([kmers])
        padded = pad_sequences(seqs, maxlen=max_lens[k], padding="post")
        encoded[f"input_k{k}"] = padded
    return encoded


def predict_confidence(model, encoded: dict,
                        k_sizes: list, true_class_idx: int) -> tuple:
    """
    Return (softmax_prob, logit_margin) for true class.
    logit_margin = true_class_logit - max_other_logit
    More sensitive than softmax for saturated high-confidence models.
    """
    input_list = [encoded[f"input_k{k}"] for k in sorted(k_sizes)]
    probs = model.predict(input_list, verbose=0)[0]
    softmax_prob = float(probs[true_class_idx])

    # Compute logit margin via log-softmax approximation
    # log(p_true) - log(max_other_p)
    log_probs   = np.log(probs + 1e-10)
    true_logit  = log_probs[true_class_idx]
    other_max   = max(log_probs[i] for i in range(len(probs))
                      if i != true_class_idx)
    logit_margin = float(true_logit - other_max)

    return softmax_prob, logit_margin


# Sequence level nucleotide masking

def mask_nucleotides(sequence: str, start_nt: int, length_nt: int) -> str:
    """
    Mask a contiguous nucleotide region by replacing its bases with 'N'.

    The masked sequence is subsequently re-encoded for both k=5
    and k=6 branches, ensuring that the perturbation affects
    both model inputs.
    """
    chars = list(sequence)
    end = min(start_nt + length_nt, len(chars))
    for i in range(start_nt, end):
        chars[i] = "N"
    return "".join(chars)


def kmer_pos_to_nt_span(kmer_pos: int, k: int, kernel: int) -> tuple:
    """
    Map a convolutional output position to its nucleotide-level
    effective receptive field (ERF).

    For k-mer size k and convolutional kernel size F, the ERF
    spans k + F - 1 nucleotides, starting at the corresponding
    k-mer position.

    Returns:
    tuple: (start_nt, length_nt)
    """
    start_nt  = kmer_pos
    length_nt = k + (kernel - 1)   # ERF = k + F - 1
    return start_nt, length_nt


def find_kmer_in_sequence(sequence: str, kmer_str: str, k: int) -> list:
    """
    Find all occurrences of a k-mer in a nucleotide sequence.

    Returns:
        list[int]: Zero-based start positions of all matches,
        including overlapping occurrences.
    """
    positions = []
    for i in range(len(sequence) - k + 1):
        if sequence[i:i + k] == kmer_str:
            positions.append(i)
    return positions



# Per-sample activation-based ERF localization


def get_top_attribution_position(model, sequence: str,
                                  tokenizers: dict, k_sizes: list,
                                  max_lens: dict,
                                  k: int = 5, kernel: int = 3) -> tuple:
    """
    Identify the highest-activation filter and its position
    within the selected convolutional branch for a given sequence.

    Returns:
        tuple: (top_kmer_string, nucleotide_start_position, erf_length)

    The position corresponds to the nucleotide-level ERF of the
    maximum activation in the selected branch.
    """
    # Build attribution sub-model
    conv_layer_names = {
        (5, 3): "conv1d",
        (5, 5): "conv1d_1",
        (6, 3): "conv1d_2",
        (6, 5): "conv1d_3",
    }
    layer_name = conv_layer_names.get((k, kernel))
    if layer_name is None:
        return None, None, None

    try:
        conv_layer = model.get_layer(layer_name)
        sub_model  = tf.keras.Model(inputs=model.inputs,
                                    outputs=conv_layer.output)
    except ValueError:
        return None, None, None

    enc  = encode_single(sequence, tokenizers, k_sizes, max_lens)
    acts = sub_model.predict(enc, verbose=0)[0]  # (L_conv, n_filters)

    # Best filter and position
    best_filter = int(np.argmax(acts.max(axis=0)))
    best_pos    = int(np.argmax(acts[:, best_filter]))

    # Map k-mer position → nucleotide span
    start_nt, length_nt = kmer_pos_to_nt_span(best_pos, k, kernel)

    # Get k-mer string at that position
    reverse_index = {v: u for u, v in tokenizers[k].word_index.items()}
    token_id  = int(enc[f"input_k{k}"][0][best_pos])
    kmer_str  = reverse_index.get(token_id, "<PAD>")

    return kmer_str, start_nt, length_nt


# Main occlusion loop

def run_occlusion(model, sequences: list, labels: list,
                  label_classes: list, tokenizers: dict,
                  max_lens: dict, k_sizes: list,
                  confidence_threshold: float = 0.9,
                  n_samples: int = 200,
                  n_random_trials: int = 5,
                  n_top_filters: int = 5,
                  attribution_k: int = 6,
                  attribution_kernel: int = 5) -> pd.DataFrame:

    k = attribution_k
    kernel = attribution_kernel
    conv_layer_names = {
        (5, 3): "conv1d",
        (5, 5): "conv1d_1",
        (6, 3): "conv1d_2",
        (6, 5): "conv1d_3",
    }
    if (k, kernel) not in conv_layer_names:
        raise ValueError(f"Unsupported attribution branch: k={k}, kernel={kernel}")
    print(f"  Attribution branch for occlusion: k={k}, kernel={kernel}")

    # Get all predictions
    print("  Encoding all sequences...")
    all_probs = []
    for seq in sequences:
        enc = encode_single(seq, tokenizers, k_sizes, max_lens)
        inp = [enc[f"input_k{ki}"] for ki in sorted(k_sizes)]
        prob = model.predict(inp, verbose=0)[0]
        all_probs.append(prob)
    all_probs = np.array(all_probs)

    pred_indices = np.argmax(all_probs, axis=1)
    pred_conf    = all_probs.max(axis=1)

    try:
        true_indices = np.array([label_classes.index(l) for l in labels])
    except ValueError as e:
        raise ValueError(f"Label mismatch: {e}")

    correct   = pred_indices == true_indices
    confident = pred_conf >= confidence_threshold
    valid     = np.where(correct & confident)[0]

    print(f"  Valid samples: {len(valid)}")

    rng = np.random.default_rng(42)
    if n_samples < len(valid):
        # Stratified sampling — equal samples per class
        from collections import defaultdict as _dd
        class_indices = _dd(list)
        for i in valid:
            class_indices[labels[i]].append(i)
        n_classes = len(class_indices)
        per_class = max(1, n_samples // n_classes)
        stratified = []
        for cls, cls_idx in class_indices.items():
            cls_arr = np.array(cls_idx)
            n_take = min(per_class, len(cls_arr))
            chosen = rng.choice(cls_arr, size=n_take, replace=False)
            stratified.extend(chosen.tolist())
        valid = np.array(stratified)
        print(f"  Stratified: {per_class} per class, {len(valid)} total")

    records = []
    print(f"  Running occlusion on {len(valid)} samples...")

    for i, idx in enumerate(valid):
        if i % 20 == 0:
            print(f"    {i}/{len(valid)}", end="\r")

        seq            = sequences[idx]
        label          = labels[idx]
        true_class_idx = label_classes.index(label)
        original_conf  = float(all_probs[idx][true_class_idx])

        # Logit margin — more sensitive than softmax for saturated models
        log_p       = np.log(all_probs[idx] + 1e-10)
        orig_margin = float(log_p[true_class_idx] -
                            max(log_p[i] for i in range(len(log_p))
                                if i != true_class_idx))

        # Get per-sample attribution position
        kmer_str, start_nt, length_nt = get_top_attribution_position(
            model, seq, tokenizers, k_sizes, max_lens, k=k, kernel=kernel
        )

        if kmer_str is None or kmer_str == "<PAD>" or start_nt is None:
            records.append({
                "sample_idx": int(idx), "class": label,
                "original_conf": original_conf,
                "delta_attributed": None, "mean_delta_random": None,
                "attribution_effect": None, "status": "attribution_failed"
            })
            continue

        # Check nucleotide span is valid
        if start_nt + length_nt > len(seq):
            records.append({
                "sample_idx": int(idx), "class": label,
                "original_conf": original_conf,
                "delta_attributed": None, "mean_delta_random": None,
                "attribution_effect": None, "status": "span_out_of_range"
            })
            continue

        # ----- Attributed masking — mask TOP-5 filters ERF spans ----------
        # Single window too small for high-confidence model → mask top-5
        try:
            conv_layer = model.get_layer(conv_layer_names[(k, kernel)])
            sub_model  = tf.keras.Model(inputs=model.inputs,
                                        outputs=conv_layer.output)
            enc_orig = encode_single(seq, tokenizers, k_sizes, max_lens)
            acts = sub_model.predict(enc_orig, verbose=0)[0]  # (L_conv, n_filters)

            # Top-5 filters by max activation
            filter_max_vals = acts.max(axis=0)
            top5_filters    = np.argsort(filter_max_vals)[::-1][:n_top_filters]
            top5_positions  = [int(np.argmax(acts[:, f])) for f in top5_filters]

            # Mask all top-5 ERF spans in sequence
            masked_seq = seq
            masked_positions = set()
            for pos in top5_positions:
                nt_start, nt_len = kmer_pos_to_nt_span(pos, k, kernel)
                for nt in range(nt_start, min(nt_start + nt_len, len(seq))):
                    masked_positions.add(nt)

            chars = list(masked_seq)
            for nt in masked_positions:
                chars[nt] = "N"
            masked_seq = "".join(chars)
            total_masked = len(masked_positions)

        except Exception:
            masked_seq   = mask_nucleotides(seq, start_nt, length_nt)
            total_masked = length_nt

        masked_enc  = encode_single(masked_seq, tokenizers, k_sizes, max_lens)
        masked_conf, masked_margin = predict_confidence(model, masked_enc,
                                         k_sizes, true_class_idx)
        delta_attr        = original_conf - masked_conf
        delta_attr_margin = orig_margin - masked_margin

        # ------ Random masking (same total nucleotide count) -------------
        seq_len = len(seq)
        random_deltas = []
        random_margin_deltas = []
        for _ in range(n_random_trials):
            if total_masked >= seq_len:
                continue
            # Random contiguous block matched by total masked nucleotide count
            max_start = seq_len - total_masked
            if max_start <= 0:
                continue
            rand_start = int(rng.integers(0, max_start))
            rand_positions = list(range(rand_start, rand_start + total_masked))
            chars = list(seq)
            for nt in rand_positions:
                chars[nt] = "N"
            rand_masked = "".join(chars)
            rand_enc    = encode_single(rand_masked, tokenizers,
                                        k_sizes, max_lens)
            rand_conf, rand_margin = predict_confidence(model, rand_enc,
                                             k_sizes, true_class_idx)
            random_deltas.append(original_conf - rand_conf)
            random_margin_deltas.append(orig_margin - rand_margin)

        mean_rand_delta = float(np.mean(random_deltas)) if random_deltas else 0.0
        attr_effect     = delta_attr - mean_rand_delta

        # ---- Motif-family occlusion -------------------------------------------
        delta_motif_margin = 0.0
        mean_motif_random  = 0.0
        n_motif_masked     = 0
        n_occurrences      = 0
        try:
            all_positions = find_kmer_in_sequence(seq, kmer_str, k)
            if all_positions and kmer_str != "<PAD>":
                motif_positions = set()
                for nt_start in all_positions:
                    for nt in range(nt_start, min(nt_start + k, len(seq))):
                        motif_positions.add(nt)
                n_motif_masked = len(motif_positions)
                n_occurrences  = len(all_positions)
                chars = list(seq)
                for nt in motif_positions:
                    chars[nt] = "N"
                motif_enc = encode_single("".join(chars), tokenizers, k_sizes, max_lens)
                _, motif_margin = predict_confidence(model, motif_enc, k_sizes, true_class_idx)
                delta_motif_margin = orig_margin - motif_margin
                motif_rand_deltas = []
                for _ in range(n_random_trials):
                    if n_motif_masked <= 0 or n_motif_masked >= seq_len:
                        continue
                    rand_pos = rng.choice(seq_len, size=n_motif_masked, replace=False)
                    chars_r = list(seq)
                    for nt in rand_pos:
                        chars_r[nt] = "N"
                    rand_enc = encode_single("".join(chars_r), tokenizers, k_sizes, max_lens)
                    _, rand_m = predict_confidence(model, rand_enc, k_sizes, true_class_idx)
                    motif_rand_deltas.append(orig_margin - rand_m)
                mean_motif_random = float(np.mean(motif_rand_deltas)) if motif_rand_deltas else 0.0
        except Exception:
            pass

        records.append({
            "sample_idx":         int(idx),
            "class":              label,
            "original_conf":      original_conf,
            "masked_conf":        masked_conf,
            "delta_attributed":   delta_attr,
            "mean_delta_random":  mean_rand_delta,
            "attribution_effect": attr_effect,
            "kmer_str":           kmer_str,
            "start_nt":           start_nt,
            "length_nt":          length_nt,
            "delta_attr_margin":   delta_attr_margin,
            "mean_margin_random":  float(np.mean(random_margin_deltas)) if random_margin_deltas else 0.0,
            "delta_motif_margin":  delta_motif_margin,
            "mean_motif_random":   mean_motif_random,
            "n_motif_occurrences": n_occurrences,
            "n_motif_masked_nt":   n_motif_masked,
            "top_anchor_kmer":     kmer_str,
            "attribution_k":       k,
            "attribution_kernel":  kernel,
            "status":             "ok",
        })

    print(f"    {len(valid)}/{len(valid)} **")
    return pd.DataFrame(records)


# Summary and plots


def summarize_and_plot(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    found = df[df["status"] == "ok"].copy()
    print(f"\n  Valid occlusion samples: {len(found)} / {len(df)}")

    if found.empty:
        print("  !!  No valid samples — check attribution pipeline.")
        return pd.DataFrame()

    summary = found.groupby("class").agg(
        n=("sample_idx", "count"),
        mean_original=("original_conf", "mean"),
        mean_delta_attr=("delta_attributed", "mean"),
        mean_delta_rand=("mean_delta_random", "mean"),
        mean_effect=("attribution_effect", "mean"),
        mean_margin_attr=("delta_attr_margin", "mean"),
        mean_margin_rand=("mean_margin_random", "mean"),
        mean_motif_margin=("delta_motif_margin", "mean"),
        mean_motif_rand=("mean_motif_random", "mean"),
        mean_n_occurrences=("n_motif_occurrences", "mean"),
    ).round(4).reset_index()
    summary["motif_effect"] = summary["mean_motif_margin"] - summary["mean_motif_rand"]
    summary["motif_stronger"] = summary["mean_motif_margin"] > summary["mean_motif_rand"]

    summary["attr_stronger_softmax"] = summary["mean_delta_attr"] > summary["mean_delta_rand"]
    summary["attr_stronger_margin"]  = summary["mean_margin_attr"] > summary["mean_margin_rand"]
    summary["margin_effect"] = summary["mean_margin_attr"] - summary["mean_margin_rand"]

    print("\n  Per-class logit margin summary:")
    print(summary[["class","mean_margin_attr","mean_margin_rand",
                   "margin_effect","attr_stronger_margin"]].to_string(index=False))

    n_stronger = summary["attr_stronger_margin"].sum()
    print(f"\n  Attribution stronger (logit margin): {n_stronger}/{len(summary)} classes")

    summary.to_csv(os.path.join(output_dir, "occlusion_summary.csv"), index=False)
    found.to_csv(os.path.join(output_dir, "occlusion_results.csv"), index=False)

    # ── Plot: 2 panels — logit margin boxplot + effect bar 
    classes = sorted(found["class"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Logit margin boxplot
    ax = axes[0]
    data_attr = [found[found["class"]==c]["delta_attr_margin"].values
                 for c in classes]
    data_rand = [found[found["class"]==c]["mean_margin_random"].values
                 for c in classes]

    positions = np.arange(len(classes))
    bp1 = ax.boxplot(data_attr,
                     positions=positions * 2 - 0.3, widths=0.5,
                     patch_artist=True,
                     boxprops=dict(facecolor="steelblue", alpha=0.7),
                     medianprops=dict(color="navy", linewidth=2))
    bp2 = ax.boxplot(data_rand,
                     positions=positions * 2 + 0.3, widths=0.5,
                     patch_artist=True,
                     boxprops=dict(facecolor="coral", alpha=0.7),
                     medianprops=dict(color="darkred", linewidth=2))
    ax.set_xticks(positions * 2)
    ax.set_xticklabels([c.replace(" ", "\n") for c in classes], fontsize=7)
    ax.set_ylabel("Δ Log-odds Margin\n(attributed − masked)")
    ax.set_title("Logit Margin Drop: Attributed vs Random Masking\n"
                 "(larger = more prediction-relevant)")
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["Attributed window", "Random window"], fontsize=8)

    # Panel 2: Margin effect bar
    ax2 = axes[1]
    colors = ["steelblue" if v > 0 else "coral"
              for v in summary["margin_effect"]]
    ax2.bar(range(len(summary)), summary["margin_effect"],
            color=colors, edgecolor="k", alpha=0.8)
    ax2.set_xticks(range(len(summary)))
    ax2.set_xticklabels([c.replace(" ", "\n") for c in summary["class"]],
                        fontsize=7)
    ax2.set_ylabel("Margin Attribution Effect\n"
                   "(attr margin drop − random margin drop)")
    ax2.set_title(f"Logit Margin Attribution Effect per Class\n"
                  f"({n_stronger}/10 classes: attributed > random)")
    ax2.axhline(0, ls="--", color="gray", lw=0.8)
    ax2.axhline(0, ls="--", color="gray", lw=0.8)

    plt.suptitle("Occlusion Validation: ERF-window Perturbation Sensitivity\n"
                 "(sequence-level N masking, both k=5 and k=6 re-encoded)",
                 fontsize=11)
    plt.tight_layout()
    if {"attribution_k", "attribution_kernel"}.issubset(found.columns):
        branch_k = int(found["attribution_k"].iloc[0])
        branch_kernel = int(found["attribution_kernel"].iloc[0])
        out = os.path.join(output_dir, f"occlusion_boxplot_k{branch_k}_ker{branch_kernel}.png")
    else:
        out = os.path.join(output_dir, "occlusion_boxplot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")

    n_stronger = summary["attr_stronger_margin"].sum()
    print(f"\n{'='*60}")
    print(f"  Attribution stronger than random (ERF-position): {n_stronger}/{len(summary)} classes")
    print(f"{'='*60}")

    if "mean_motif_margin" in summary.columns:
        n_motif_stronger = summary["motif_stronger"].sum()
        print(f"\n  Motif-family occlusion summary:")
        print(summary[["class","mean_motif_margin","mean_motif_rand",
                       "motif_effect","mean_n_occurrences","motif_stronger"]].to_string(index=False))
        print(f"\n  Motif-family stronger than random: {n_motif_stronger}/{len(summary)} classes")

    return summary


# Main

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    k_sizes = [5, 6]

    print("\n[1/5] Loading model...")
    model = tf.keras.models.load_model(args.model_path)

    print("\n[2/5] Loading artefacts...")
    max_lens      = get_max_lens_from_model(model, k_sizes)
    tokenizers    = load_tokenizers(args.tokenizer_dir, k_sizes)
    label_classes = load_label_classes(args.label_classes)

    print("\n[3/5] Loading data...")
    df        = pd.read_csv(args.data_path)
    sequences = df["Sequence"].tolist()
    labels    = df["Organism_Name"].tolist()

    if args.n_samples and args.n_samples < len(sequences):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(sequences), args.n_samples, replace=False)
        sequences = [sequences[i] for i in idx]
        labels    = [labels[i]    for i in idx]
        print(f"  Subsampled to {args.n_samples} sequences.")

    print("\n[4/5] Running occlusion validation...")
    df_occ = run_occlusion(
        model=model,
        sequences=sequences,
        labels=labels,
        label_classes=label_classes,
        tokenizers=tokenizers,
        max_lens=max_lens,
        k_sizes=k_sizes,
        confidence_threshold=args.confidence,
        n_samples=args.n_samples or len(sequences),
        n_random_trials=args.n_random,
        n_top_filters=args.n_top_filters,
        attribution_k=args.attribution_k,
        attribution_kernel=args.attribution_kernel,
    )

    print("\n[5/5] Summarising and plotting...")
    summarize_and_plot(df_occ, args.output_dir)
    print(f"\n** Done. Results in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence-level occlusion analysis for DiCNN-UniK ERF attribution"
    )

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--label_classes", required=True)
    parser.add_argument("--data_path", required=True)

    parser.add_argument("--output_dir", default="occlusion_results/")
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_random", type=int, default=15)
    parser.add_argument("--n_top_filters", type=int, default=5)

    parser.add_argument(
        "--attribution_k",
        type=int,
        default=6,
        choices=[5, 6],
        help="K-mer size of the selected attribution branch (default: 6)"
    )

    parser.add_argument(
        "--attribution_kernel",
        type=int,
        default=5,
        choices=[3, 5],
        help="Convolutional kernel size of the selected branch (default: 5)"
    )

    args = parser.parse_args()
    main(args)