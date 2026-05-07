from __future__ import annotations

import pickle
import tarfile
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = Path("dataset")
CIFAR_ARCHIVE = DATA_DIR / "cifar-10-python.tar.gz"
CIFAR_EXTRACTED = DATA_DIR / "cifar-10-batches-py"
CAT_LABEL = 3


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, x, y):
    m = x.shape[1]
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    a = np.clip(a, 1e-10, 1 - 1e-10)
    cost = -(1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    dz = a - y
    dw = (1 / m) * np.dot(x, dz.T)
    db = (1 / m) * np.sum(dz)
    gradients = {"dw": dw, "db": db}
    return gradients, float(cost)


def optimize(w, b, x, y, num_iterations, learning_rate):
    costs = []
    for i in range(num_iterations):
        gradients, cost = propagate(w, b, x, y)
        dw = gradients["dw"]
        db = gradients["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            print(f"iteration {i:4d} | cost {cost:.4f}")

    params = {"w": w, "b": b}
    gradients = {"dw": dw, "db": db}
    return params, gradients, costs


def predict(w, b, x):
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    return (a > 0.5).astype(int)


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.001):
    n_x = X_train.shape[0]
    w, b = initialize(n_x)
    params, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate
    )
    w = params["w"]
    b = params["b"]
    Y_pred_test = predict(w, b, X_test)
    Y_pred_train = predict(w, b, X_train)

    train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100

    print(f"train accuracy: {train_acc:.2f}%")
    print(f"test accuracy:  {test_acc:.2f}%")

    return {
        "w": w,
        "b": b,
        "costs": costs,
        "grads": grads,
        "Y_pred_train": Y_pred_train,
        "Y_pred_test": Y_pred_test,
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
    }


def download_cifar10():
    DATA_DIR.mkdir(exist_ok=True)
    if not CIFAR_ARCHIVE.exists():
        print(f"downloading dataset to {CIFAR_ARCHIVE} ...")
        urllib.request.urlretrieve(DATA_URL, CIFAR_ARCHIVE)
    if not CIFAR_EXTRACTED.exists():
        print(f"extracting dataset to {DATA_DIR} ...")
        with tarfile.open(CIFAR_ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA_DIR)
    return CIFAR_EXTRACTED


def load_cifar_batch(batch_path: Path):
    with batch_path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="dtype\\(\\): align should be passed.*")
            batch = pickle.load(f, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch["labels"], dtype=np.int64)
    return images, labels


def load_cifar10(extracted_dir: Path):
    train_images = []
    train_labels = []
    for i in range(1, 6):
        images, labels = load_cifar_batch(extracted_dir / f"data_batch_{i}")
        train_images.append(images)
        train_labels.append(labels)

    X_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    X_test, y_test = load_cifar_batch(extracted_dir / "test_batch")
    return X_train, y_train, X_test, y_test


def sample_balanced_binary_dataset(
    images,
    labels,
    positive_label=CAT_LABEL,
    positive_count=1000,
    negative_count=1000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(labels == positive_label)
    negative_idx = np.flatnonzero(labels != positive_label)

    if len(positive_idx) < positive_count or len(negative_idx) < negative_count:
        raise ValueError("not enough samples to create a balanced dataset")

    chosen_pos = rng.choice(positive_idx, size=positive_count, replace=False)
    chosen_neg = rng.choice(negative_idx, size=negative_count, replace=False)
    indices = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(indices)

    sampled_images = images[indices]
    sampled_labels = (labels[indices] == positive_label).astype(np.int64)
    return sampled_images, sampled_labels


def preprocess_images(images, image_size=32):
    processed = np.empty((len(images), image_size, image_size, 3), dtype=np.float32)
    for idx, image in enumerate(images):
        pil_image = Image.fromarray(image)
        if image_size != 32:
            pil_image = pil_image.resize((image_size, image_size))
        processed[idx] = np.asarray(pil_image, dtype=np.float32) / 255.0

    flattened = processed.reshape(processed.shape[0], -1).T
    return flattened


def prepare_dataset(
    train_positive=1000,
    train_negative=1000,
    test_positive=400,
    test_negative=400,
    image_size=32,
):
    extracted_dir = download_cifar10()
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_cifar10(extracted_dir)

    train_images, train_labels = sample_balanced_binary_dataset(
        X_train_raw,
        y_train_raw,
        positive_count=train_positive,
        negative_count=train_negative,
        seed=42,
    )
    test_images, test_labels = sample_balanced_binary_dataset(
        X_test_raw,
        y_test_raw,
        positive_count=test_positive,
        negative_count=test_negative,
        seed=123,
    )

    X_train = preprocess_images(train_images, image_size=image_size)
    X_test = preprocess_images(test_images, image_size=image_size)
    Y_train = train_labels.reshape(1, -1)
    Y_test = test_labels.reshape(1, -1)

    print(f"train set: X={X_train.shape}, Y={Y_train.shape}")
    print(f"test set:  X={X_test.shape}, Y={Y_test.shape}")

    return X_train, Y_train, X_test, Y_test


def export_cifar_to_image_folders(extracted_dir: Path, export_root: Path = DATA_DIR):
    cat_dir = export_root / "cat"
    not_cat_dir = export_root / "not_cat"
    cat_dir.mkdir(parents=True, exist_ok=True)
    not_cat_dir.mkdir(parents=True, exist_ok=True)

    existing_cat = any(cat_dir.iterdir())
    existing_not_cat = any(not_cat_dir.iterdir())
    if existing_cat or existing_not_cat:
        print(f"image folders already populated: {cat_dir} and {not_cat_dir}")
        return

    split_files = [("train", f"data_batch_{i}") for i in range(1, 6)]
    split_files.append(("test", "test_batch"))

    cat_count = 0
    not_cat_count = 0
    for split_name, batch_name in split_files:
        images, labels = load_cifar_batch(extracted_dir / batch_name)
        for idx, (image, label) in enumerate(zip(images, labels)):
            class_name = "cat" if label == CAT_LABEL else "not_cat"
            target_dir = cat_dir if class_name == "cat" else not_cat_dir
            target_path = target_dir / f"{split_name}_{batch_name}_{idx:05d}_label{label}.png"
            Image.fromarray(image).save(target_path)
            if class_name == "cat":
                cat_count += 1
            else:
                not_cat_count += 1

    print(f"exported {cat_count} cat images to {cat_dir}")
    print(f"exported {not_cat_count} not-cat images to {not_cat_dir}")


def plot_costs(costs, output_path="training_cost.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(costs)) * 100, costs, linewidth=2)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("Training Cost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved cost curve to {output_path}")


def main():
    extracted_dir = download_cifar10()
    export_cifar_to_image_folders(extracted_dir)
    X_train, Y_train, X_test, Y_test = prepare_dataset(
        train_positive=1000,
        train_negative=1000,
        test_positive=400,
        test_negative=400,
        image_size=32,
    )
    results = model(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_iterations=2000,
        learning_rate=0.001,
    )
    plot_costs(results["costs"])


if __name__ == "__main__":
    main()
