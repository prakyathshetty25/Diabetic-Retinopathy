"""
PyTorch Fine-Tuning Script for Retinal DR Classifier on Real Project Dataset.
Dataset Path: /home/prakyath/Desktop/diabetic/Diagnosis of Diabetic Retinopathy
Saves fine-tuned weights to backend/app/weights/dr_resnet50_weights.pth.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

# Ensure backend directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.model import RetinalDRClassifier
from app.config import IMAGENET_MEAN, IMAGENET_STD

DATASET_ROOT = "/home/prakyath/Desktop/diabetic/Diagnosis of Diabetic Retinopathy"


class ProjectDRDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", is_training: bool = True):
        self.split_dir = os.path.join(root_dir, split)
        self.samples = []

        no_dr_dir = os.path.join(self.split_dir, "No_DR")
        dr_dir = os.path.join(self.split_dir, "DR")

        if os.path.exists(no_dr_dir):
            for fname in os.listdir(no_dr_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(no_dr_dir, fname), 0))

        if os.path.exists(dr_dir):
            for fname in os.listdir(dr_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Assign DR sample to Class 2 (Moderate DR) as base DR target
                    self.samples.append((os.path.join(dr_dir, fname), 2))

        if is_training:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        try:
            pil_img = Image.open(img_path).convert("RGB")
            tensor_img = self.transform(pil_img)
            return tensor_img, target
        except Exception:
            return torch.zeros((3, 224, 224)), target


def train_model():
    print("==================================================")
    print("TRAINING RESNET50 ON PROJECT DATASET")
    print(f"Dataset Path: {DATASET_ROOT}")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")

    train_dataset = ProjectDRDataset(DATASET_ROOT, split="train", is_training=True)
    valid_dataset = ProjectDRDataset(DATASET_ROOT, split="valid", is_training=False)

    print(f"Train Samples: {len(train_dataset)} | Val Samples: {len(valid_dataset)}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = RetinalDRClassifier(backbone_name="resnet50", pretrained=True, num_classes=5).to(device)

    # Fine-tune layer4 and classification head
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.backbone.layer4.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    epochs = 6
    best_val_acc = 0.0

    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "weights"))
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "dr_resnet50_weights.pth")

    print(f"\nStarting fine-tuning for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_targets in valid_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_targets)

                val_loss += v_loss.item() * val_inputs.size(0)
                v_preds = val_outputs.argmax(dim=1)
                val_correct += (v_preds == val_targets).sum().item()
                val_total += val_inputs.size(0)

        val_loss_epoch = val_loss / max(1, val_total)
        val_acc_epoch = (val_correct / max(1, val_total)) * 100.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss_epoch:.4f} | Val Acc: {val_acc_epoch:.2f}%")

        if val_acc_epoch >= best_val_acc:
            best_val_acc = val_acc_epoch
            torch.save(model.state_dict(), weights_path)
            print(f" -> Saved best model weights to {weights_path} (Val Acc: {val_acc_epoch:.2f}%)")

    print("\n==================================================")
    print(f" TRAINING COMPLETE! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f" Saved Model Weights: {weights_path}")
    print("==================================================")


if __name__ == "__main__":
    train_model()
