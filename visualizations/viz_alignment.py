import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
from tqdm import tqdm
from dataloader import get_dataloaders
from models import load_clap, get_audio_embeddings, get_text_embeddings, concepts as DEFAULT_CONCEPTS


@torch.no_grad()
def collect_class_embeddings(train_loader, clap_model, processor, device, n_classes):
    class_embs = {i: [] for i in range(n_classes)}
    for batch in tqdm(train_loader, desc="collecting audio embeddings"):
        wavs = [w.numpy() for w in batch["waveform"]]
        embs = get_audio_embeddings(clap_model, processor, wavs, device)  # B x d
        for emb, label in zip(embs, batch["label"]):
            class_embs[int(label)].append(emb.cpu())

    return {k: torch.stack(v, dim=0) for k, v in class_embs.items()}


@torch.no_grad()
def compute_alignment(class_embs, clap_model, processor, class_names, device, concepts=None):
    if concepts is None:
        concepts = {k: DEFAULT_CONCEPTS.get(k, [f"a person speaking with {k}"]) for k in class_names}

    concept_list, per_class_slices = [], {}
    for name in class_names:
        start = len(concept_list)
        concept_list.extend(concepts[name])
        per_class_slices[name] = (start, len(concept_list))

    text_embs = get_text_embeddings(clap_model, processor, concept_list, device)  # N_C x d

    n_cls = len(class_names)
    N_C = len(concept_list)
    alignment = np.zeros((n_cls, N_C))

    for i, name in enumerate(class_names):
        embs = class_embs[i].to(device)      

        # Mean audio embedding for this class
        mean_emb = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)  # 1 x d
        sims = (mean_emb @ text_embs.T).squeeze(0).cpu().numpy() # N_C
        alignment[i] = sims

    return alignment, concept_list, per_class_slices


def plot_alignment(alignment, class_names, concept_list, per_class_slices, save_path):
    n_cls = len(class_names)
    ncols = min(4, n_cls)
    nrows = int(np.ceil(n_cls / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5.5, nrows * 3.8),
                              squeeze=False)
    axes_flat = axes.flatten()

    # global colour range across all concepts for consistent colour encoding
    vmin = alignment.min() * 0.95
    vmax = alignment.max() * 1.05
    cmap = cm_module.get_cmap("RdYlGn")

    for i, name in enumerate(class_names):
        ax = axes_flat[i]
        start, end = per_class_slices[name]

        # concepts belonging to this class
        concepts_here = concept_list[start:end]
        scores = alignment[i, start:end]

        # sort descending by alignment score
        order  = np.argsort(scores)[::-1]
        scores = scores[order]
        labels = [concepts_here[j] for j in order]

        short_labels = [" ".join(l.split()[:7]) for l in labels]
        colors = [cmap((s - vmin) / max(vmax - vmin, 1e-6)) for s in scores]

        bars = ax.barh(range(len(scores)), scores, color=colors, edgecolor="none", height=0.7)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=6.5)

        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Mean cosine similarity", fontsize=8)
        ax.set_title(f"{name}  (concepts in class block)", fontsize=9, fontweight="bold")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(axis="x", alpha=0.25)

        chance = alignment[i].mean()
        ax.axvline(chance, color="steelblue", linewidth=0.9, linestyle=":",
                    label=f"mean over all concepts ({chance:.3f})")
        ax.legend(fontsize=6, loc="lower right")

    for j in range(n_cls, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Concept–class alignment: mean cos(audio, text) per concept\n"
        "Blue dotted = average sim over all concepts (flat = no discrimination)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved at {save_path}")
    plt.close()

def print_alignment_stats(alignment, class_names, concept_list, per_class_slices):
    print("\nConcept discrimination per class (max − min alignment within class block):")
    for i, name in enumerate(class_names):
        start, end = per_class_slices[name]
        block = alignment[i, start:end]
        disc = block.max() - block.min()
        top_c = concept_list[start + int(block.argmax())]
        print(f"{name:15s}  disc={disc:.4f}  best='{' '.join(top_c.split()[:8])}'")

    print("\nCross-class alignment (how discriminative is each concept across all classes):")
    for c_idx, concept in enumerate(concept_list):
        class_sims = alignment[:, c_idx]
        disc = class_sims.max() - class_sims.min()
        best_class = class_names[int(class_sims.argmax())]
        if disc > 0.03:   
            print(f"  disc={disc:.3f}  best_class={best_class:12s}  '{' '.join(concept.split()[:8])}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="figs/concept_alignment.png")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class names. Defaults to DEFAULT_CONCEPTS keys.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    train_loader, _, _, class_names_dataset = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    if args.classes:
        class_names = [c.strip() for c in args.classes.split(",")]
    else:
        class_names = class_names_dataset   
        print(f"Classes from dataset: {class_names}")

    clap_model, processor = load_clap(device)
    class_embs = collect_class_embeddings(train_loader, clap_model, processor, device, n_classes=len(class_names))

    alignment, concept_list, per_class_slices = compute_alignment(class_embs, clap_model, processor, class_names, device)

    print_alignment_stats(alignment, class_names, concept_list, per_class_slices)
    plot_alignment(alignment, class_names, concept_list, per_class_slices, args.save_path)

if __name__ == "__main__":
    main()