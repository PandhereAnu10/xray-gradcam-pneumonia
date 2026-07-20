import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs("/kaggle/working/outputs", exist_ok=True)

# This matches the ACTUAL attached dataset structure (confirmed via os.walk):
#   /kaggle/input/datasets/organizations/nih-chest-xrays/data/images_XXX/images/*.png
#   /kaggle/input/datasets/organizations/nih-chest-xrays/data/Data_Entry_2017.csv
NIH_ROOT = "/kaggle/input/datasets/organizations/nih-chest-xrays/data"


# built a filename - full path index, then loads labels
print("Indexing image files across all images_XXX/images folders (takes ~10-20s)...")
file_index = {}
for path in glob.glob(os.path.join(NIH_ROOT, "images_*", "images", "*")):
    file_index[os.path.basename(path)] = path
print(f"Indexed {len(file_index):,} image files")

csv_path = os.path.join(NIH_ROOT, "Data_Entry_2017.csv")
df = pd.read_csv(csv_path)
print(f"CSV rows: {len(df):,}, columns: {list(df.columns)}")

# Binary task: does "Pneumonia" appear anywhere in this image's Finding Labels?
df["has_pneumonia"] = df["Finding Labels"].str.contains("Pneumonia").astype(int)
df["filepath"] = df["Image Index"].map(file_index)
df = df.dropna(subset=["filepath"])  # keep only rows whose image is actually in this session's mounted images_XXX folders

print(f"Rows with an available image file: {len(df):,}")
print(f"Positive (pneumonia) rows available: {df['has_pneumonia'].sum():,}")

# NIH ChestX-ray14 is heavily imbalanced (~1-2% pneumonia across the full set).
# For a fast, balanced demo, took every available positive and an equal-ish
# number of randomly sampled negatives, rather than training on the raw
# imbalance (which would need a much longer run + class weighting to matter).
pos_df = df[df["has_pneumonia"] == 1]
neg_df = df[df["has_pneumonia"] == 0].sample(n=min(len(pos_df) * 3, len(df)), random_state=0)
balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=0).reset_index(drop=True)
print(f"Balanced subset: {len(balanced_df):,} images "
      f"({balanced_df['has_pneumonia'].sum():,} positive, {(balanced_df['has_pneumonia']==0).sum():,} negative)")

train_df, test_df = train_test_split(
    balanced_df, test_size=0.2, stratify=balanced_df["has_pneumonia"], random_state=0
)
print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

CLASSES = ["No Pneumonia", "Pneumonia"]


# a small Dataset class reading straight from the indexed paths
IMG_SIZE = 224
NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats, we're using pretrained weights

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class NIHXrayDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)
        label = int(row["has_pneumonia"])
        return img, label


train_ds = NIHXrayDataset(train_df, train_tf)
test_ds = NIHXrayDataset(test_df, test_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)


# model transfer learning from a pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


# train
EPOCHS = 3  # this dataset is easy enough that 3 epochs of fine-tuning goes a long way

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    print(f"epoch {epoch+1}/{EPOCHS}  loss={running_loss/total:.4f}  train_acc={correct/total:.3f}")


# evaluate + confusion matrix diagram
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

print("\n" + classification_report(all_labels, all_preds, target_names=CLASSES))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix: Pneumonia Classifier")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("/kaggle/working/outputs/confusion_matrix.png", dpi=150)
plt.show()


# Grad-CAM on correct & incorrect predictions
cam_engine = GradCAM(model, model.layer4[-1])

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization to get a displayable uint8 image."""
    img = tensor.clone().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(NORM_STD) + np.array(NORM_MEAN)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def make_gradcam_panel(indices: list[int], filename: str, title: str) -> None:
    tiles = []
    for idx in indices:
        x, y_true = test_ds[idx]
        x_batch = x.unsqueeze(0).to(DEVICE)
        x_batch.requires_grad_(True)
        pred = model(x_batch).argmax(1).item()

        cam = cam_engine(x_batch, class_idx=pred)
        img_uint8 = denormalize(x)
        overlay = overlay_heatmap(img_uint8, cam)
        tiles.append(overlay)

    grid = np.concatenate(tiles, axis=1)
    plt.figure(figsize=(4 * len(indices), 4))
    plt.imshow(grid)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/outputs/{filename}", dpi=150)
    plt.show()


correct_idx = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p == l][:4]
wrong_idx = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l][:4]

print(f"\nFound {len(wrong_idx)} misclassified examples in the first pass -- showing up to 4 of each.")

make_gradcam_panel(correct_idx, "gradcam_correct.png",
                    "Grad-CAM: correctly classified X-rays (where is the model looking?)")
if wrong_idx:
    make_gradcam_panel(wrong_idx, "gradcam_incorrect.png",
                        "Grad-CAM: MISCLASSIFIED X-rays (is the model looking somewhere odd?)")
else:
    print("No misclassified examples in this run -- try a stricter validation split or fewer epochs "
          "to surface some errors worth visualizing; a model that's never wrong makes for a less "
          "interesting article than one where you can show a genuine failure case.")
