import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset, sigmoid


def initial_params(n_x, n_h, n_y, seed=None):
    if seed is not None:
        np.random.seed(seed)

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    params = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }
    return params


def forward(X, params):
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]

    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
    }
    return a2, cache


def compute_cost(a2, Y):
    m = Y.shape[1]
    eps = 1e-8
    cost = (-1 / m) * np.sum(Y * np.log(a2 + eps) + (1 - Y) * np.log(1 - a2 + eps))
    return cost


def backward(X, Y, params, cache):
    m = X.shape[1]

    w2 = params["w2"]
    a1 = cache["a1"]
    a2 = cache["a2"]

    dz2 = a2 - Y
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * (1 - np.power(a1, 2))
    dw1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    grads = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }
    return grads


def update(params, grads, learning_rate):
    params["w1"] = params["w1"] - learning_rate * grads["dw1"]
    params["b1"] = params["b1"] - learning_rate * grads["db1"]
    params["w2"] = params["w2"] - learning_rate * grads["dw2"]
    params["b2"] = params["b2"] - learning_rate * grads["db2"]
    return params


def train_model(X, Y, n_h, learning_rate=0.5, num_iters=5000, seed=None, print_cost=False):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    params = initial_params(n_x, n_h, n_y, seed=seed)

    for i in range(num_iters):
        a2, cache = forward(X, params)
        cost = compute_cost(a2, Y)
        grads = backward(X, Y, params, cache)
        params = update(params, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print(f"iter={i}, cost={cost:.6f}")

    return params


def predict(X, params):
    a2, _ = forward(X, params)
    Y_pred = (a2 > 0.5).astype(int)
    return Y_pred


def compute_accuracy(X, Y, params):
    Y_pred = predict(X, params)
    acc = np.mean(Y_pred.ravel() == Y.ravel())
    return acc


def split_dataset(X, Y, train_ratio=0.8, seed=0):
    m = X.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)

    train_size = int(m * train_ratio)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]
    X_test = X[:, test_idx]
    Y_test = Y[:, test_idx]

    return X_train, Y_train, X_test, Y_test


def run_one_setting(m, n_h, learning_rate=0.5, num_iters=5000, data_seed=1, split_seed=0, init_seed=0):
    X, Y = load_planar_dataset(m=m)
    X_train, Y_train, X_test, Y_test = split_dataset(X, Y, train_ratio=0.8, seed=split_seed)

    params = train_model(
        X_train,
        Y_train,
        n_h=n_h,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=init_seed,
        print_cost=False,
    )

    train_acc = compute_accuracy(X_train, Y_train, params)
    test_acc = compute_accuracy(X_test, Y_test, params)

    return train_acc, test_acc


def run_grid(m_list, n_h_list, learning_rate=0.5, num_iters=5000, num_runs=3):
    train_map = np.zeros((len(m_list), len(n_h_list)))
    test_map = np.zeros((len(m_list), len(n_h_list)))

    for i, m in enumerate(m_list):
        for j, n_h in enumerate(n_h_list):
            train_scores = []
            test_scores = []

            for run in range(num_runs):
                train_acc, test_acc = run_one_setting(
                    m=m,
                    n_h=n_h,
                    learning_rate=learning_rate,
                    num_iters=num_iters,
                    data_seed=1,
                    split_seed=run,
                    init_seed=run,
                )
                train_scores.append(train_acc)
                test_scores.append(test_acc)

            train_map[i, j] = np.mean(train_scores)
            test_map[i, j] = np.mean(test_scores)

            print(
                f"m={m:4d}, n_h={n_h:3d}, "
                f"train={train_map[i, j]:.3f}, test={test_map[i, j]:.3f}"
            )

    return train_map, test_map


def plot_heatmap(values, x_labels, y_labels, title):
    plt.figure(figsize=(8, 5))
    plt.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="accuracy")
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xlabel("n_h")
    plt.ylabel("m")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    m_list = [50, 100, 200, 400, 800]
    n_h_list = [1, 2, 4, 8, 16, 32]

    train_map, test_map = run_grid(
        m_list=m_list,
        n_h_list=n_h_list,
        learning_rate=0.5,
        num_iters=5000,
        num_runs=3,
    )

    plot_heatmap(train_map, n_h_list, m_list, "Train Accuracy Map")
    plot_heatmap(test_map, n_h_list, m_list, "Test Accuracy Map")
