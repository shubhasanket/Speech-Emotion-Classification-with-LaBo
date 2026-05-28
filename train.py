from dataloader import get_dataloaders
from models import CLAPProbe, E2ECNN, LaBoAudio, load_clap, get_audio_embeddings
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import torch
torch.backends.cudnn.enabled = False  

os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"


def get_input(batch, model_name, clap_model, processor, device):
    if model_name == "e2e":
        return batch["mel"].to(device)
    else:
        # pre-compute frozen CLAP audio embeddings for this batch
        wavs = [w.numpy() for w in batch["waveform"]]  
        return get_audio_embeddings(clap_model, processor, wavs, device)


def train(model, model_name, train_loader, val_loader, clap_model, processor, device, args):
    
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}"):
            x = get_input(batch, model_name, clap_model, processor, device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, model_name, val_loader, clap_model, processor, device)
        print(f"Epoch {epoch} | loss: {total_loss/len(train_loader):.4f} | val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.save_dir}/{model_name}.pt")
            print(f"Saved checkpoint | val acc={best_val_acc:.4f}")


@torch.no_grad()
def evaluate(model, model_name, loader, clap_model, processor, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x= get_input(batch, model_name, clap_model, processor, device)
        labels = batch["label"].to(device)
        preds= model(x).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total+= labels.size(0)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="clap_probe", choices=["clap_probe", "e2e", "labo"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    n_classes = len(class_names)

    clap_model, processor = load_clap(device)
    if args.model == "clap_probe":
        model = CLAPProbe(n_classes=n_classes).to(device)
    elif args.model == "e2e":
        model = E2ECNN(n_classes=n_classes).to(device)
        clap_model, processor = None, None  # not needed for CNN model
    elif args.model == "labo":
        model = LaBoAudio(class_names, clap_model, processor, device).to(device)

    train(model, args.model, train_loader, val_loader, clap_model, processor, device, args)

    model.load_state_dict(torch.load(f"{args.save_dir}/{args.model}.pt", map_location=device))
    test_acc = evaluate(model, args.model, test_loader, clap_model, processor, device)
    print(f"\nTest accuracy ({args.model}): {test_acc:.4f}")

    if args.model == "labo":
        print("\nTop concepts per class:")
        for i, name in enumerate(class_names):
            top = model.top_concepts(i, k=3)
            print(f"  {name}: {top}")



if __name__ == "__main__":
    main()
