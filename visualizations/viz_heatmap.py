import argparse
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from models import concepts as DEFAULT_CONCEPTS
import os


def load_w_matrix(checkpoint_path, device="cpu"):
    state = torch.load(checkpoint_path, map_location=device)
    W = state["W"] # (n_cls, N_C)
    return W


def build_concept_meta(class_names, concepts=None):
    if concepts is None:
        concepts = {k: DEFAULT_CONCEPTS.get(k, [f"a person speaking with {k}"]) for k in class_names}

    concept_list, concept_labels = [], []
    per_class_slices = {}
    for i, name in enumerate(class_names):
        start = len(concept_list)
        for c in concepts[name]:
            concept_list.append(c)
            concept_labels.append(i)
        per_class_slices[name] = (start, len(concept_list))

    return concept_list, concept_labels, per_class_slices

def softmax_w(W):
    return F.softmax(W, dim=0).detach().cpu().numpy()   # (n_cls, N_C)

def plot_heatmap(W_norm, class_names, concept_list, per_class_slices, save_path):
    n_cls, N_C = W_norm.shape
    short_labels = [" ".join(c.split()[:6]) for c in concept_list]

    fig, ax = plt.subplots(figsize=(max(14, N_C * 0.22), max(4, n_cls * 0.55)))
    im = ax.imshow(W_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xticks(range(N_C))
    ax.set_xticklabels(short_labels, rotation=75, ha="right", fontsize=6.5)
    ax.set_xlabel("Concepts", fontsize=10)
    ax.set_ylabel("Emotion class", fontsize=10)
    ax.set_title("LaBo W matrix (softmax weights)\nRows = classes · Columns = concepts", fontsize=11)

    for name in list(per_class_slices.keys())[1:]:
        boundary = per_class_slices[name][0] - 0.5
        ax.axvline(boundary, color="white", linewidth=1.5, linestyle="--", alpha=0.7)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = [
        (per_class_slices[n][0] + per_class_slices[n][1] - 1) / 2
        for n in class_names
    ]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(class_names, fontsize=8, rotation=30, ha="left")
    ax2.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Softmax weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap → {save_path}")
    plt.close()


def print_top_concepts(W_norm, class_names, concept_list, k=5):
    print("\nTop concepts per class (by softmax weight):")
    for i, name in enumerate(class_names):
        weights = W_norm[i]
        top_idx = np.argsort(weights)[::-1][:k]
        print(f"\n  {name.upper()}")
        for rank, idx in enumerate(top_idx, 1):
            print(f"{rank}. [{weights[idx]:.4f}]  {concept_list[idx]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LaBo .pt checkpoint (state_dict with key 'W')")
    parser.add_argument("--save-path", type=str, default="figs/w_heatmap.png")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top concepts to print per class")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names, e.g. angry,happy,sad,neutral")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    device = "cpu"

    # Infer class names
    if args.classes:
        class_names = [c.strip() for c in args.classes.split(",")]
    else:
        class_names = sorted(DEFAULT_CONCEPTS.keys())
        print(f"Using default class list: {class_names}")

    concept_list, concept_labels, per_class_slices = build_concept_meta(class_names)

    W_raw = load_w_matrix(args.checkpoint, device)
    assert W_raw.shape == (len(class_names), len(concept_list)), (
        f"Shape mismatch: got {tuple(W_raw.shape)}, "
        f"expected ({len(class_names)}, {len(concept_list)})"
    )

    W_norm = softmax_w(W_raw)

    print_top_concepts(W_norm, class_names, concept_list, k=args.top_k)
    plot_heatmap(W_norm, class_names, concept_list, per_class_slices, args.save_path)


if __name__ == "__main__":
    main()