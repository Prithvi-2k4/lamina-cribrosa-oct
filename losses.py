"""Dataset and patient-based 80/20 split."""
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from config import IMAGE_ROOT, LABEL_ROOT, IMG_SIZE, SEED


def collect_pairs(image_root: Path = IMAGE_ROOT,
                  label_root: Path = LABEL_ROOT):
    """Walk image_root for *.png, expect a matching *.jpg at the same
    relative path under label_root. Patient ID = first folder under image_root.
    Returns (train_pairs, val_pairs) lists of dicts."""
    image_root, label_root = Path(image_root), Path(label_root)
    all_pairs = []
    for img_path in image_root.rglob("*.png"):
        rel        = img_path.relative_to(image_root)
        patient_id = rel.parts[0]
        label_path = label_root / rel.with_suffix(".jpg")
        if label_path.exists():
            all_pairs.append({"image": img_path,
                              "label": label_path,
                              "patient": patient_id})

    print(f"Found {len(all_pairs)} matched image-label pairs.")

    patients = sorted({p["patient"] for p in all_pairs})
    train_pats, val_pats = train_test_split(
        patients, test_size=0.2, random_state=SEED, shuffle=True)

    train_pairs = [p for p in all_pairs if p["patient"] in train_pats]
    val_pairs   = [p for p in all_pairs if p["patient"] in val_pats]
    print(f"Patients: {len(patients)} total → "
          f"{len(train_pats)} train / {len(val_pats)} val")
    print(f"Images:   {len(train_pairs)} train / {len(val_pairs)} val")
    return train_pairs, val_pairs


class OCTDataset(Dataset):
    """Loads (image, mask), applies SAME random aug to both."""
    def __init__(self, pairs, augment: bool = False, img_size: int = IMG_SIZE):
        self.pairs    = pairs
        self.augment  = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p   = self.pairs[idx]
        img  = cv2.imread(str(p["image"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(p["label"]), cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,  (self.img_size, self.img_size),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)
            if random.random() > 0.5:
                angle = random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D(
                    (self.img_size/2, self.img_size/2), angle, 1.0)
                img  = cv2.warpAffine(img,  M, (self.img_size, self.img_size),
                                      flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size),
                                      flags=cv2.INTER_NEAREST)
            if random.random() > 0.5:
                img = np.clip(img.astype(np.float32) *
                              random.uniform(0.9, 1.1), 0, 255).astype(np.uint8)

        img_t  = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t
