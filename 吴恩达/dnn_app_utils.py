from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from assignment3 import (
    L_model_backward,
    L_model_forward,
    compute_cost,
    initialize_parameters,
    initialize_parameters_deep,
    linear_activation_backward,
    linear_activation_forward,
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    update_parameters,
)


CLASSES = np.array([b"non-cat", b"cat"])
DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def _read_images(paths, label, size=(64, 64)):
    images = []
    labels = []

    for path in paths:
        with Image.open(path) as img:
            image = img.convert("RGB").resize(size)
            images.append(np.array(image))
            labels.append(label)

    return images, labels


def load_data(train_per_class=105, test_per_class=25, seed=1):
    rng = np.random.default_rng(seed)

    cat_paths = np.array(sorted((DATASET_DIR / "cat").glob("*.png")), dtype=object)
    not_cat_paths = np.array(sorted((DATASET_DIR / "not_cat").glob("*.png")), dtype=object)

    if len(cat_paths) == 0 or len(not_cat_paths) == 0:
        raise FileNotFoundError("dataset/cat or dataset/not_cat is empty")

    cat_idx = rng.permutation(len(cat_paths))
    not_cat_idx = rng.permutation(len(not_cat_paths))

    train_cat_idx = cat_idx[:train_per_class]
    test_cat_idx = cat_idx[train_per_class : train_per_class + test_per_class]
    train_not_cat_idx = not_cat_idx[:train_per_class]
    test_not_cat_idx = not_cat_idx[train_per_class : train_per_class + test_per_class]

    train_cat_x, train_cat_y = _read_images(cat_paths[train_cat_idx], 1)
    train_not_cat_x, train_not_cat_y = _read_images(not_cat_paths[train_not_cat_idx], 0)
    test_cat_x, test_cat_y = _read_images(cat_paths[test_cat_idx], 1)
    test_not_cat_x, test_not_cat_y = _read_images(not_cat_paths[test_not_cat_idx], 0)

    train_x = train_cat_x + train_not_cat_x
    train_y = train_cat_y + train_not_cat_y
    test_x = test_cat_x + test_not_cat_x
    test_y = test_cat_y + test_not_cat_y

    train_perm = rng.permutation(len(train_x))
    test_perm = rng.permutation(len(test_x))

    train_x = np.array(train_x, dtype=np.uint8)[train_perm]
    train_y = np.array(train_y, dtype=np.uint8)[train_perm].reshape(1, -1)
    test_x = np.array(test_x, dtype=np.uint8)[test_perm]
    test_y = np.array(test_y, dtype=np.uint8)[test_perm].reshape(1, -1)

    return train_x, train_y, test_x, test_y, CLASSES


def predict(X, y, parameters):
    probs, _ = L_model_forward(X, parameters)
    predictions = (probs > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print("Accuracy: " + str(accuracy))
    return predictions


def print_mislabeled_images(classes, X, y, p, limit=10):
    mislabeled_indices = np.where(p != y)[1]

    if len(mislabeled_indices) == 0:
        print("No mislabeled images.")
        return

    limit = min(limit, len(mislabeled_indices))
    plt.figure(figsize=(12, 4))

    for i, idx in enumerate(mislabeled_indices[:limit]):
        plt.subplot(2, (limit + 1) // 2, i + 1)
        image = X[:, idx].reshape(64, 64, 3)
        plt.imshow(image)
        plt.axis("off")
        true_label = classes[int(y[0, idx])].decode("utf-8")
        pred_label = classes[int(p[0, idx])].decode("utf-8")
        plt.title(f"p:{pred_label}\ny:{true_label}")

    plt.tight_layout()
