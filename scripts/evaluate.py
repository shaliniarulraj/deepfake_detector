import os
import argparse
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

def load_model(ckpt, device):
    # Load checkpoint
    ck = torch.load(ckpt, map_location=device)

    # Build model architecture exactly as used in training
    model = models.resnet18(weights=None)
    in_f = model.fc.in_features
    model.fc = torch.nn.Linear(in_f, 2)  # Match original training arch (no Dropout)

    # Load weights
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)

    return model, ck["classes"], ck["img_size"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/best_resnet18.pt")
    ap.add_argument("--data_root", type=str, default="data/frames")
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    device = torch.device("cpu")

    # Load model + metadata
    model, classes, img_size = load_model(args.ckpt, device)

    # Define transforms
    tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Dataset & DataLoader
    ds = datasets.ImageFolder(os.path.join(args.data_root, args.split), transform=tfms)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    # Evaluation loop
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"Accuracy {acc:.4f} | Precision {p:.4f} | Recall {r:.4f} | F1 {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print(classification_report(labels, preds, target_names=classes, zero_division=0))
