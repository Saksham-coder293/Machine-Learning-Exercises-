"""
Exercise 12 (Géron, Hands-On ML, Ch.4 — Training Models):
Implement batch gradient descent with early stopping for softmax regression,
using only NumPy (no Scikit-Learn estimators), on the iris dataset.

Only sklearn.datasets.load_iris is used to fetch the data itself — the
actual model (softmax, cross-entropy loss, gradients, gradient descent,
early stopping) is all hand-rolled in NumPy.
"""

import numpy as np
from sklearn.datasets import load_iris


# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
def load_and_prepare_iris(random_state=42):
    iris = load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]           # 3 classes: 0, 1, 2

    # add bias term (x0 = 1) to every instance
    X_with_bias = np.c_[np.ones(len(X)), X]

    rng = np.random.default_rng(random_state)
    n = len(X_with_bias)
    shuffled_indices = rng.permutation(n)

    test_ratio, val_ratio = 0.2, 0.2
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size

    train_idx = shuffled_indices[:train_size]
    val_idx = shuffled_indices[train_size:train_size + val_size]
    test_idx = shuffled_indices[train_size + val_size:]

    X_train, y_train = X_with_bias[train_idx], y[train_idx]
    X_valid, y_valid = X_with_bias[val_idx], y[val_idx]
    X_test, y_test = X_with_bias[test_idx], y[test_idx]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def to_one_hot(y, n_classes=None):
    n_classes = n_classes or (y.max() + 1)
    m = len(y)
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot


# ---------------------------------------------------------------------------
# 2. Softmax
# ---------------------------------------------------------------------------
def softmax(logits):
    # subtract row-max for numerical stability (avoids overflow in exp)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 3. Training loop: batch gradient descent + early stopping + optional L2
# ---------------------------------------------------------------------------
def train_softmax_regression(
    X_train, y_train, X_valid, y_valid,
    n_classes=3, eta=0.5, n_epochs=5001,
    epsilon=1e-7, alpha=0.001, patience=None, verbose_every=500,
):
    """
    eta      : learning rate
    epsilon  : small constant added inside log() to avoid log(0)
    alpha    : L2 regularization strength (set to 0 for no regularization)
    patience : if set, stop after this many consecutive epochs without
               validation-loss improvement (in addition to best-Theta
               early stopping)
    """
    n_features = X_train.shape[1]
    Y_train_one_hot = to_one_hot(y_train, n_classes)
    Y_valid_one_hot = to_one_hot(y_valid, n_classes)

    rng = np.random.default_rng(42)
    Theta = rng.standard_normal((n_features, n_classes))

    m = len(X_train)
    best_loss = np.inf
    best_epoch = 0
    best_Theta = Theta.copy()
    epochs_no_improve = 0
    history = []

    for epoch in range(n_epochs):
        logits = X_train @ Theta
        Y_proba = softmax(logits)

        # cross-entropy loss with L2 regularization (bias term excluded)
        xentropy_loss = -np.mean(
            np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1)
        )
        l2_loss = 0.5 * alpha * np.sum(np.square(Theta[1:]))
        loss = xentropy_loss + l2_loss

        error = Y_proba - Y_train_one_hot
        gradients = (1 / m) * X_train.T @ error
        gradients += np.r_[np.zeros((1, n_classes)), alpha * Theta[1:]]

        Theta = Theta - eta * gradients

        # --- validation pass (no parameter update, just for monitoring) ---
        logits_valid = X_valid @ Theta
        Y_proba_valid = softmax(logits_valid)
        xentropy_valid = -np.mean(
            np.sum(Y_valid_one_hot * np.log(Y_proba_valid + epsilon), axis=1)
        )
        l2_valid = 0.5 * alpha * np.sum(np.square(Theta[1:]))
        valid_loss = xentropy_valid + l2_valid

        history.append((epoch, loss, valid_loss))

        if epoch % verbose_every == 0:
            print(f"epoch {epoch:5d} - train loss {loss:.5f} - valid loss {valid_loss:.5f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_Theta = Theta.copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best epoch was {best_epoch}, best valid loss {best_loss:.5f})")
                break

    return best_Theta, history, best_epoch, best_loss


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def predict(X, Theta):
    logits = X @ Theta
    Y_proba = softmax(logits)
    return np.argmax(Y_proba, axis=1)


def accuracy(X, y, Theta):
    y_pred = predict(X, Theta)
    return np.mean(y_pred == y)


# ---------------------------------------------------------------------------
# 5. Run it all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_prepare_iris()

    print("Shapes:", X_train.shape, X_valid.shape, X_test.shape)

    best_Theta, history, best_epoch, best_loss = train_softmax_regression(
        X_train, y_train, X_valid, y_valid,
        eta=0.5, n_epochs=5001, alpha=0.001, patience=None,
    )

    val_acc = accuracy(X_valid, y_valid, best_Theta)
    test_acc = accuracy(X_test, y_test, best_Theta)
    print(f"\nBest epoch: {best_epoch}, best validation loss: {best_loss:.5f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nLearned Theta (rows=[bias, petal_length, petal_width], cols=classes):")
    print(best_Theta)
