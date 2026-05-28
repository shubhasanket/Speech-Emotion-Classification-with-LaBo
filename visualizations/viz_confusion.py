import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import get_dataloaders
from models import CLAPProbe, load_clap, get_audio_embeddings, get_text_embeddings, concepts as DEFAULT_CONCEPTS

@torch.no_grad()
def predict_probe(model, loader, clap_model, processor, device):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="probe"):
        wavs = [w.numpy() for w in batch["waveform"]]
        emb = get_audio_embeddings(clap_model, processor, wavs, device)
        p = model(emb).argmax(dim=-1).cpu()
        preds.extend(p.numpy())
        labels.extend(batch["label"].numpy())
    return np.array(preds), np.array(labels)


@torch.no_grad()
def predict_labo(model, loader, clap_model, processor, device):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="labo"):
        wavs = [w.numpy() for w in batch["waveform"]]
        emb = get_audio_embeddings(clap_model, processor, wavs, device)
        p = model(emb).argmax(dim=-1).cpu()
        preds.extend(p.numpy())
        labels.extend(batch["label"].numpy())
    return np.array(preds), np.array(labels)


@torch.no_grad()
def predict_zero_shot(loader, class_names, clap_model, processor, device, agg="mean"):
    if agg == "single":
        prompts = [f"a person speaking with {n} emotion" for n in class_names]
        text_emb = get_text_embeddings(clap_model, processor, prompts, device)
    else:  # mean / max
        concepts = {k: DEFAULT_CONCEPTS.get(k, [f"a person speaking with {k}"]) for k in class_names}
        concept_embs = []
        for n in class_names:
            e = get_text_embeddings(clap_model, processor, concepts[n], device)
            concept_embs.append(e)

    preds, labels = [], []
    for batch in tqdm(loader, desc=f"zero-shot ({agg})"):
        wavs  = [w.numpy() for w in batch["waveform"]]
        audio = get_audio_embeddings(clap_model, processor, wavs, device)
        if agg == "single":
            logits = audio @ text_emb.T
            p = logits.argmax(dim=-1).cpu()
        elif agg == "mean":
            scores = torch.stack([audio @ e.T.mean(dim=1) for e in concept_embs], dim=1)
            p = scores.argmax(dim=-1).cpu()
        elif agg == "max":
            scores = torch.stack([(audio @ e.T).max(dim=-1).values for e in concept_embs], dim=1)
            p = scores.argmax(dim=-1).cpu()

        preds.extend(p.numpy())
        labels.extend(batch["label"].numpy())
    return np.array(preds), np.array(labels)


@torch.no_grad()
def predict_knn(train_loader, test_loader, clap_model, processor, device, k=5):

    def collect(loader, desc):
        embs, labs = [], []
        for batch in tqdm(loader, desc=desc):
            wavs = [w.numpy() for w in batch["waveform"]]
            e = get_audio_embeddings(clap_model, processor, wavs, device)
            embs.append(e.cpu())
            labs.append(batch["label"])
        return torch.cat(embs), torch.cat(labs)

    train_embs, train_labels = collect(train_loader, "knn-train")
    test_embs, test_labels = collect(test_loader,  "knn-test")

    sims = test_embs @ train_embs.T
    topk = sims.topk(k, dim=-1).indices
    preds = train_labels[topk].mode(dim=-1).values

    return preds.numpy(), test_labels.numpy()

def compute_cm(preds, labels, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm

def plot_cm(ax, cm, class_names, title, acc):
    n = len(class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(f"{title}\nacc = {acc:.3f}", fontsize=9)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    
    return im

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labo-ckpt", type=str, default=None)
    parser.add_argument("--probe-ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--zero-shot-agg", type=str, default="mean", choices=["single", "mean", "max"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="figs/confusion_matrices.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    train_loader, _, test_loader, class_names = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    n_classes = len(class_names)

    clap_model, processor = load_clap(device)

    results = {} 
    if args.labo_ckpt and os.path.exists(args.labo_ckpt):
        from models import LaBoAudio
        labo = LaBoAudio(class_names, clap_model, processor, device).to(device)
        labo.load_state_dict(torch.load(args.labo_ckpt, map_location=device))
        p, l = predict_labo(labo, test_loader, clap_model, processor, device)
        results["LaBo"] = (p, l)
        del labo
    else:
        print("LaBo checkpoint not found — skipping.")

    if args.probe_ckpt and os.path.exists(args.probe_ckpt):
        probe = CLAPProbe(n_classes=n_classes).to(device)
        probe.load_state_dict(torch.load(args.probe_ckpt, map_location=device))
        p, l = predict_probe(probe, test_loader, clap_model, processor, device)
        results["CLAP Probe"] = (p, l)
        del probe
    else:
        print("CLAP Probe checkpoint not found — skipping.")

    p, l = predict_knn(train_loader, test_loader, clap_model, processor, device, k=args.knn_k)
    results[f"kNN (k={args.knn_k})"] = (p, l)

    p, l = predict_zero_shot(test_loader, class_names, clap_model, processor, device,
                              agg=args.zero_shot_agg)
    results[f"Zero-shot ({args.zero_shot_agg})"] = (p, l)

    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots,
                              figsize=(n_classes * 1.05 * n_plots, n_classes * 1.05 + 1.5),
                              squeeze=False)

    last_im = None
    for ax, (name, (preds, labels)) in zip(axes[0], results.items()):
        cm  = compute_cm(preds, labels, n_classes)
        acc = (preds == labels).mean()
        last_im = plot_cm(ax, cm, class_names, name, acc)

    fig.suptitle("Confusion matrices (recall-normalised)\nRows = true class, Columns = predicted", fontsize=11, y=1.02)
    fig.colorbar(last_im, ax=axes[0], fraction=0.02, pad=0.02, label="Recall fraction")
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.save_path}")

    for name, (preds, labels) in results.items():
        cm  = compute_cm(preds, labels, n_classes)
        acc = (preds == labels).mean()
        print(f"\n{name}  (overall acc={acc:.4f})")
        for i, cn in enumerate(class_names):
            recall = cm[i, i] / max(cm[i].sum(), 1)
            print(f"  {cn:15s}  recall={recall:.3f}  ({cm[i, i]}/{cm[i].sum()})")

if __name__ == "__main__":
    main()