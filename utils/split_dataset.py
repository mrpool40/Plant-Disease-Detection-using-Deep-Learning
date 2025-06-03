import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0

    classes = os.listdir(source_dir)
    for cls in tqdm(classes, desc="Splitting dataset"):
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, split_images in splits.items():
            split_dir = os.path.join(dest_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_dir, img)
                try:
                    shutil.copyfile(src, dst)
                except Exception as e:
                    print(f"⚠️ Skipping file: {src}\n   Reason: {e}")

if __name__ == '__main__':
    SOURCE_DIR = '../dataset/PlantVillage'   # relative to utils
    DEST_DIR = '../dataset/plantvillage'     # where train/val/test will be created

    split_dataset(SOURCE_DIR, DEST_DIR)
    print("✅ Dataset split into train / val / test.")
