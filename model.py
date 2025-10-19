"""
resnet_extract_and_softmax_train.py

- Extracts features using a pre-trained ResNet-18 model.
- This is a powerful deep learning approach, expected to perform much better than HOG.
- The rest of the pipeline (Softmax training, saving model) remains the same.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- THAY ĐỔI: Sử dụng PyTorch và Torchvision cho ResNet ---
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# ---------------- CONFIG ----------------
DATA_ROOT = r"D:\HOMEWORK\dataset_1\dataset"  # <-- Thay đổi đường dẫn tới thư mục dataset của bạn
SPLITS = ["train\phone", "val\phone", "test\phone"]

LABEL_MAP = {
    "non-phone": 0,
    "defective": 1,
    "non-defective": 2
}

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- THAY ĐỔI: Cấu hình cho ResNet ---
BATCH_SIZE = 32  # Có thể tăng batch size vì ResNet nhẹ hơn ViT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DO_AUGMENT = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Siêu tham số cho Softmax (bắt đầu lại với các giá trị mặc định)
LR = 0.01
LAMBDA = 0.01
EPOCHS = 100  # ResNet hội tụ nhanh hơn

# ---------------- ResNet Model Setup ----------------
print("Device:", DEVICE)
print("Loading pre-trained ResNet-18 model...")

# Tải mô hình ResNet-18 đã được huấn luyện trước trên ImageNet
weights = ResNet18_Weights.IMAGENET1K_V1
transform_for_model = weights.transforms()
model = resnet18(weights=weights).to(DEVICE)

# --- THAY ĐỔI: Loại bỏ lớp phân loại cuối cùng của ResNet ---
# Lớp cuối cùng của ResNet có tên là 'fc' (fully connected)
# Chúng ta thay thế nó bằng một lớp Identity để lấy vector đặc trưng 512 chiều
FEATURE_DIM = model.fc.in_features  # Sẽ là 512
model.fc = nn.Identity()

model.eval()
print(f"ResNet-18 loaded. Feature dimension: {FEATURE_DIM}")


# ---------------- Helpers (Không thay đổi) ----------------

def find_class_dirs(root_split_dir):
    found = []
    for dirpath, dirnames, filenames in os.walk(root_split_dir):
        basename = os.path.basename(dirpath)
        if basename in LABEL_MAP:
            found.append((basename, dirpath))
    return found


def list_images_in_dir(dir_path):
    filepaths = []
    if not os.path.isdir(dir_path):
        return filepaths
    for root, _, filenames in os.walk(dir_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTS:
                filepaths.append(os.path.join(root, fname))
    return filepaths


def load_image_rgb(path):
    # Đảm bảo ảnh được chuyển sang 3 kênh RGB
    return Image.open(path).convert("RGB")


# ---------------- Feature Extraction (Sử dụng ResNet) ----------------

def extract_features_from_paths(paths):
    if len(paths) == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    all_feats = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Extracting Features"):
        batch_paths = paths[i:i + BATCH_SIZE]
        imgs = [load_image_rgb(p) for p in batch_paths]

        inputs = [transform_for_model(img) for img in imgs]
        batch_tensor = torch.stack(inputs, dim=0).to(DEVICE)

        with torch.no_grad():
            out = model(batch_tensor)
            feats = out.cpu().numpy()
        all_feats.append(feats)

    return np.vstack(all_feats)


def build_dataset_for_split(split):
    split_dir = os.path.join(DATA_ROOT, split)
    class_dirs = find_class_dirs(split_dir)

    X_list, y_list = [], []
    for class_name, class_folder in class_dirs:
        label = LABEL_MAP[class_name]
        print(f"Processing class '{class_name}' -> label {label}")
        filepaths = list_images_in_dir(class_folder)
        if not filepaths: continue

        # Ảnh gốc
        feats_orig = extract_features_from_paths(filepaths)
        X_list.append(feats_orig)
        y_list.append(np.full(feats_orig.shape[0], label, dtype=np.int64))

        # Augmentation (dùng transform của PyTorch)
        if DO_AUGMENT:
            aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ])
            # Tạo một danh sách các ảnh đã được augment
            aug_imgs = [aug_transform(load_image_rgb(p)) for p in filepaths]

            # Trích xuất đặc trưng từ các ảnh đã augment
            all_feats_aug = []
            for i in tqdm(range(0, len(aug_imgs), BATCH_SIZE), desc="Augmenting Features"):
                batch_imgs = aug_imgs[i:i + BATCH_SIZE]
                inputs = [transform_for_model(img) for img in batch_imgs]  # Áp dụng transform chuẩn của ResNet
                batch_tensor = torch.stack(inputs, dim=0).to(DEVICE)
                with torch.no_grad():
                    out = model(batch_tensor)
                    feats = out.cpu().numpy()
                all_feats_aug.append(feats)

            if all_feats_aug:
                feats_aug = np.vstack(all_feats_aug)
                X_list.append(feats_aug)
                y_list.append(np.full(feats_aug.shape[0], label, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------- Softmax (NumPy) training (Không thay đổi) ----------------

def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def compute_loss(X, y, W, b, lambd):
    m = X.shape[0]
    scores = X @ W + b
    probs = softmax_np(scores)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(m), y] = 1
    ce = -np.sum(y_onehot * np.log(probs + 1e-12)) / m
    reg = (lambd / (2 * m)) * np.sum(W ** 2)
    return ce + reg


def compute_acc(X, y, W, b):
    scores = X @ W + b
    probs = softmax_np(scores)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)


def train_softmax(X_train, y_train, X_val, y_val, X_test, y_test, lr=0.01, lambd=0.01, epochs=200):
    m, n = X_train.shape
    n_classes = len(np.unique(y_train))
    W = np.zeros((n, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)
    for epoch in range(1, epochs + 1):
        scores = X_train @ W + b
        probs = softmax_np(scores)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(m), y_train] = 1
        dW = (X_train.T @ (probs - y_onehot)) / m + (lambd / m) * W
        db = np.mean(probs - y_onehot, axis=0)
        W -= lr * dW
        b -= lr * db
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train_loss = compute_loss(X_train, y_train, W, b, lambd)
            val_loss = compute_loss(X_val, y_val, W, b, lambd)
            test_loss = compute_loss(X_test, y_test, W, b, lambd)
            train_acc = compute_acc(X_train, y_train, W, b)
            val_acc = compute_acc(X_val, y_val, W, b)
            test_acc = compute_acc(X_test, y_test, W, b)
            print(f"Epoch {epoch} | TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, TestLoss={test_loss:.4f} | "
                  f"TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}")
    return W, b


def plot_confusion_matrix(y_true, y_pred, classes=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    feats = {}
    for split in SPLITS:
        safe_split_name = split.replace(os.path.sep, '_')
        out_x = os.path.join(OUT_DIR, f"X_{safe_split_name}_resnet.npy")
        out_y = os.path.join(OUT_DIR, f"y_{safe_split_name}_resnet.npy")
        if os.path.exists(out_x) and os.path.exists(out_y):
            print(f"[{split}] loading cached ResNet features...")
            X = np.load(out_x)
            y = np.load(out_y)
        else:
            print(f"[{split}] extracting ResNet features...")
            X, y = build_dataset_for_split(split)
            np.save(out_x, X)
            np.save(out_y, y)
        print(f"  shapes X={X.shape}, y={y.shape}")
        feats[split] = (X, y)

    X_train, y_train = feats["train\phone"]
    X_val, y_val = feats["val\phone"]
    X_test, y_test = feats["test\phone"]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-12
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    W, b = train_softmax(X_train_std.astype(np.float32), y_train.astype(np.int64),
                         X_val_std.astype(np.float32), y_val.astype(np.int64),
                         X_test_std.astype(np.float32), y_test.astype(np.int64),
                         lr=LR, lambd=LAMBDA, epochs=EPOCHS)

    model_dict = {"W": W, "b": b, "mean": mean, "std": std, "label_map": LABEL_MAP}
    model_path = os.path.join(OUT_DIR, "softmax_model_resnet.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"Saved ResNet softmax model to {model_path}")

    preds_train = np.argmax(softmax_np(X_train_std @ W + b), axis=1)
    preds_val = np.argmax(softmax_np(X_val_std @ W + b), axis=1)
    preds_test = np.argmax(softmax_np(X_test_std @ W + b), axis=1)

    class_names = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
    plot_confusion_matrix(y_train, preds_train, classes=class_names, title="ResNet Train Confusion Matrix")
    plot_confusion_matrix(y_val, preds_val, classes=class_names, title="ResNet Val Confusion Matrix")
    plot_confusion_matrix(y_test, preds_test, classes=class_names, title="ResNet Test Confusion Matrix")

    print("Done.")
