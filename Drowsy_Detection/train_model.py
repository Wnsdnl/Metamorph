import os
import cv2
# Haar Cascade for full-face detection (used before landmark overlay)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import mediapipe as mp
mp_facemesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
from model import DrowsyEyeNet, device

# Constants
IMG_SIZE = 145
BATCH_SIZE = 32
EPOCHS = 70
DATA_DIR = os.path.join(os.path.dirname(__file__), 'preprocessed_data')  # train_data/drowsy, train_data/notdrowsy

# Eye landmark indices from MediaPipe FaceMesh
LEFT_EYE_INDEXES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
all_chosen_idxs = LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES

class EyeCropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []  # list of (image_path, label)
        classes = ['drowsy', 'notdrowsy']  # folder names
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), label))
        # FaceMesh will be initialized in each worker to avoid pickling issues
        self.face_mesh = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack path and label
        img_path, label = self.samples[idx]
        # Load preprocessed image (overlay + resize already applied)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        crop_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            sample = self.transform(crop_rgb)
        else:
            sample = torch.from_numpy(crop_rgb).permute(2,0,1).float().div(255.0)
        return sample, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Transforms (augment + normalize)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # [0,1] 스케일
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Prepare dataset and split with separate transforms
    base_dataset = EyeCropDataset(DATA_DIR, transform=None)
    dataset_size = len(base_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))

    # Shuffle indices
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create train and val subsets with appropriate transforms
    train_dataset = Subset(
        EyeCropDataset(DATA_DIR, transform=train_transform),
        train_indices
    )
    val_dataset = Subset(
        EyeCropDataset(DATA_DIR, transform=val_transform),
        val_indices
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,    # 데이터 로딩을 4개의 병렬 워커로
        pin_memory=True   # GPU 전송을 위한 고정 메모리 사용
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model, loss, optimizer
    model = DrowsyEyeNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / len(train_dataset)
        train_acc  = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= len(val_dataset)
        val_acc  = val_correct / val_total

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pth'))