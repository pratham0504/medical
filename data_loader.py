import os
import warnings
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, datasets
from PIL import Image


warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    category=Warning,
)


RAW_DATA_ROOT = "data/MNIST/raw"

# CIFAR-10 image folder (PNG images stored in cifar-10/train/)
CIFAR10_IMG_DIR = os.path.join(RAW_DATA_ROOT, "cifar-10", "train")

CIFAR10_DIR_CANDIDATES = [
    os.path.join(RAW_DATA_ROOT, "cifar-10", "train"),
    os.path.join(RAW_DATA_ROOT, "CIFAR-10-dataset", "train"),
    os.path.join(RAW_DATA_ROOT, "cifar10", "train"),
]

CIFAR100_DIR_CANDIDATES = [
    os.path.join(RAW_DATA_ROOT, "CIFAR-100-dataset", "train"),
    os.path.join(RAW_DATA_ROOT, "cifar-100", "train"),
    os.path.join(RAW_DATA_ROOT, "cifar100", "train"),
]


DATASET_CANDIDATES = [
    ("HeadCT", ["HeadCT", "headct", "HandCT", "handct"]),
    ("ChestCT", ["ChestCT", "chestct"]),
    ("AbdomenCT", ["AbdomenCT", "abdomenct"]),
    ("BreastMRI", ["BreastMRI", "breastmri", "BrestMRI", "brestmri"]),
    ("Hand", ["Hand", "hand"]),
]


def _find_existing_dir(root_dir, candidates):
    if not os.path.exists(root_dir):
        return None

    existing = {entry.lower(): entry for entry in os.listdir(root_dir)}
    for candidate in candidates:
        matched = existing.get(candidate.lower())
        if matched is not None:
            return os.path.join(root_dir, matched)
    return None


def get_available_medical_datasets(root_dir=RAW_DATA_ROOT):
    datasets = []
    for display_name, candidates in DATASET_CANDIDATES:
        dataset_dir = _find_existing_dir(root_dir, candidates)
        if dataset_dir is not None:
            datasets.append((display_name, dataset_dir))
    return datasets


def _looks_like_image(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg"))


def _has_class_subdirs(root_dir):
    if not os.path.isdir(root_dir):
        return False
    for entry in os.listdir(root_dir):
        sub = os.path.join(root_dir, entry)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if _looks_like_image(f):
                    return True
    return False


def _find_existing_path(candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def get_available_class_names(root_dir=RAW_DATA_ROOT):
    """Returns fixed 7 classes: 5 medical + CIFAR10 + CIFAR100 (single-label each)."""
    class_names = [name for name, _ in get_available_medical_datasets(root_dir)]

    cifar10_dir = _find_existing_path(CIFAR10_DIR_CANDIDATES)
    if cifar10_dir is not None:
        class_names.append("CIFAR10")

    cifar100_dir = _find_existing_path(CIFAR100_DIR_CANDIDATES)
    if cifar100_dir is not None:
        class_names.append("CIFAR100")

    return class_names


class LabelOffsetDataset(Dataset):
    def __init__(self, base_dataset, offset):
        self.base_dataset = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        return image, target + self.offset


class MedicalDataset(Dataset):
    def __init__(self, root_dir, label, dataset_name, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.dataset_name = dataset_name
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(root_dir) else []

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None: image = np.zeros((64, 64), dtype=np.uint8)
        image = Image.fromarray(image)
        if self.transform: image = self.transform(image)
        return image, self.label


class FolderTreeSingleLabelDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.image_paths = []

        if os.path.isdir(root_dir):
            for current_root, _, files in os.walk(root_dir):
                for filename in files:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(current_root, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((64, 64), dtype=np.uint8)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, self.label

def load_medical_partitions(config):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    available_datasets = get_available_medical_datasets()
    names = []
    all_datasets = []

    # Medical image datasets
    for dataset_name, dataset_dir in available_datasets:
        label = len(names)
        names.append(dataset_name)
        all_datasets.append(MedicalDataset(dataset_dir, label, dataset_name, transform))

    max_samples = getattr(config, "max_cifar_samples", 5000)

    # CIFAR-10 image dataset (single-label)
    cifar10_dir = _find_existing_path(CIFAR10_DIR_CANDIDATES)
    if cifar10_dir is not None:
        label = len(names)
        names.append("CIFAR10")
        if _has_class_subdirs(cifar10_dir):
            cifar10_ds = FolderTreeSingleLabelDataset(cifar10_dir, label, transform)
        else:
            cifar10_ds = MedicalDataset(cifar10_dir, label, "CIFAR10", transform)

        if len(cifar10_ds) > max_samples:
            idx = torch.randperm(len(cifar10_ds))[:max_samples].tolist()
            all_datasets.append(Subset(cifar10_ds, idx))
        else:
            all_datasets.append(cifar10_ds)

    # CIFAR-100 image dataset (single-label)
    cifar100_dir = _find_existing_path(CIFAR100_DIR_CANDIDATES)
    if cifar100_dir is not None:
        label = len(names)
        names.append("CIFAR100")
        cifar100_ds = FolderTreeSingleLabelDataset(cifar100_dir, label, transform)
        if len(cifar100_ds) > max_samples:
            idx = torch.randperm(len(cifar100_ds))[:max_samples].tolist()
            all_datasets.append(Subset(cifar100_ds, idx))
        else:
            all_datasets.append(cifar100_ds)

    if not all_datasets:
        raise ValueError("No supported datasets found under data/MNIST/raw")

    full_dataset = ConcatDataset(all_datasets)
    
    indices = torch.randperm(len(full_dataset)).tolist()
    data_per_client = max(1, len(full_dataset) // config.num_clients)

    train_loaders = []
    for i in range(config.num_clients):
        start = i * data_per_client
        end = len(full_dataset) if i == config.num_clients - 1 else min((i + 1) * data_per_client, len(full_dataset))
        subset = Subset(full_dataset, indices[start:end])
        train_loaders.append(DataLoader(subset, batch_size=config.batch_size, shuffle=True, num_workers=0))

    test_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    return train_loaders, test_loader, names