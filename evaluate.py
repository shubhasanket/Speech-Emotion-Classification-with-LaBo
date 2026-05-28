from dataloader import get_dataloaders
from models import CLAPProbe, load_clap, get_audio_embeddings, get_text_embeddings, concepts as DEFAULT_CONCEPTS
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np

import torch
torch.backends.cudnn.enabled = False


@torch.no_grad()
def collect_embeddings(clap_model, processor, loader, device):

    all_embs, all_labels = [], []
    for batch in tqdm(loader, disable=True):
        wavs = [w.numpy() for w in batch["waveform"]]
        embs = get_audio_embeddings(clap_model, processor, wavs, device)
        all_embs.append(embs.cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_embs), torch.cat(all_labels)

    
@torch.no_grad()
def run_zero_shot(test_loader, class_names, clap_model, processor, device, args):
    if args.zero_shot_agg == "single":

        prompts = [f"a person speaking with {name} emotion" for name in class_names]
        text_emb = get_text_embeddings(clap_model, processor, prompts, device)

    else:
        concepts = {k: DEFAULT_CONCEPTS.get(k, [f"a person speaking with {k}"]) for k in class_names}
        concept_embs = []

        for name in class_names:
            c_list = concepts[name]
            emb = get_text_embeddings(clap_model, processor, c_list, device)
            concept_embs.append(emb)

    all_preds, all_labels = [], []

    for batch in tqdm(test_loader, disable=True):
        wavs = [w.numpy() for w in batch["waveform"]]
        audio = get_audio_embeddings(clap_model, processor, wavs, device)
        labels = batch["label"]

        if args.zero_shot_agg == "single":
            logits = audio @ text_emb.T
            preds = logits.argmax(dim=-1).cpu()

        elif args.zero_shot_agg == "mean":
            class_scores = torch.stack([audio @ emb.T.mean(dim=1) for emb in concept_embs], dim=1)

            preds = class_scores.argmax(dim=-1).cpu()

        elif args.zero_shot_agg == "max":
            class_scores = torch.stack([(audio @ emb.T).max(dim=-1).values for emb in concept_embs], dim=1)
            preds = class_scores.argmax(dim=-1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds== all_labels).mean()

    return acc

@torch.no_grad()
def run_knn(train_loader, test_loader, clap_model, processor, device, args): #
    train_embs, train_labels = collect_embeddings(clap_model, processor, train_loader, device)
    test_embs, test_labels = collect_embeddings(clap_model, processor, test_loader, device)


    sims = test_embs @ train_embs.T
    topk = sims.topk(args.knn_k, dim=-1).indices
    preds = train_labels[topk].mode(dim=-1).values

    all_preds = preds.numpy()
    all_labels = test_labels.numpy()

    acc = (all_preds == all_labels).mean()

    return acc




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "knn"])
    parser.add_argument("--zero-shot-agg", type=str, default="mean", choices=["single", "mean", "max", "all"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, _, test_loader, class_names = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    clap_model, processor = load_clap(device)

    if args.mode == "zero_shot":
        if args.zero_shot_agg == "all":
            for agg in ["single", "mean", "max"]:
                args.zero_shot_agg = agg
                acc = run_zero_shot(test_loader, class_names, clap_model, processor, device, args)
                print(f"{agg}: {acc:.4f}")
        else:
            acc = run_zero_shot(test_loader, class_names, clap_model, processor, device, args)
            print(f"{acc:.4f}")
    else:
        acc = run_knn(train_loader, test_loader, clap_model, processor, device, args)
        print(f"{acc:.4f}")



if __name__ == "__main__":
    main()
