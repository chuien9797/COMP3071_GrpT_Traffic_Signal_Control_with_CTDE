import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

from model import VGG16Custom
from evaluator import Evaluator

# ========================== CONFIG ==========================

BASE_DIR = '/content/drive/MyDrive/All_Models/Multiclass_Classification3'
CHECKPOINT_PATH = '/content/drive/MyDrive/All_Models/examples/For_Testing/best_model_B_with_scloss.pth'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN_NAME = 'final_test'
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================== DATASET ===========================

mean = [0.6366578, 0.5400036, 0.6245009]
std = [0.12276892, 0.15195936, 0.10576411]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_dataset = ImageFolder(os.path.join(BASE_DIR, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
class_names = test_dataset.classes
n_classes = len(class_names)

# ========================= MODEL ============================

model = VGG16Custom(num_classes=n_classes, in_channels=3, pretrained=False).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["net"])
model.eval()

# ======================= EVALUATOR ==========================

evaluator = Evaluator(BASE_DIR, split="test")

# ========================== TEST ============================

def test_model(model, test_loader, evaluator, device, n_classes, class_names):
    model.eval()
    y_score = torch.tensor([]).to(device)
    all_preds = []
    all_labels = []
    total_loss = []
    all_img_paths = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(targets.cpu().numpy().tolist())
            y_score = torch.cat((y_score, probs), 0)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score_np = y_score.detach().cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    try:
        auc = roc_auc_score(y_true, y_score_np, multi_class='ovr')
    except:
        auc = -1

    # === Save CSV
    df_preds = pd.DataFrame({
        "ImageIndex": list(range(len(y_true))),
        "ActualLabel": [class_names[y] for y in y_true],
        "PredictedLabel": [class_names[p] for p in y_pred]
    })
    df_preds.to_csv(os.path.join(SAVE_DIR, "predictions.csv"), index=False)
    print(f"✅ Saved predictions to predictions.csv")

    # === Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
    plt.close()
    print("✅ Saved confusion matrix to confusion_matrix.png")

    # === Per-Class Precision / Recall / F1
    print("\n📊 Classification Report (per class):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # === Print Summary
    print(f"\n📈 Accuracy:  {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"🔁 Recall:    {rec:.4f}")
    print(f"📊 F1-Score:  {f1:.4f}")
    print(f"🏆 Average ROC AUC: {auc:.4f}")

    # === Save evaluator outputs
    evaluator.evaluate(y_score_np, save_folder=SAVE_DIR, run=RUN_NAME)

    return acc, prec, rec, f1, auc

# ========================== RUN =============================

if __name__ == '__main__':
    test_model(model, test_loader, evaluator, DEVICE, n_classes, class_names)
