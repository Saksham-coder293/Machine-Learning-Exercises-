"""
Solutions to the end-of-chapter exercises (Géron, Hands-On ML, Ch.5 — SVMs)

Covers:
  9.  LinearSVC vs SVC(kernel='linear') vs SGDClassifier on sklearn's wine dataset
  10. Multiclass SVM classifier on the wine dataset (one-vs-rest under the hood),
      with feature scaling + hyperparameter tuning
  11. SVR (tuned via RandomizedSearchCV) on the California housing dataset
"""

import numpy as np
from sklearn.datasets import load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import reciprocal, uniform


# ---------------------------------------------------------------------------
# 9. LinearSVC on the wine dataset (linearly separable-ish, 3 classes)
# ---------------------------------------------------------------------------
def exercise_9(random_state=42):
    wine = load_wine()
    X, y = wine["data"], wine["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # LinearSVC (like all SVM-based models) is sensitive to feature scale —
    # wine's features span wildly different ranges (e.g. proline ~300-1700
    # vs. hue ~0.5-1.7), so scaling is essential here, not optional.
    lin_svc = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1, max_iter=10_000, random_state=random_state),
    )
    lin_svc.fit(X_train, y_train)
    lin_svc_acc = accuracy_score(y_test, lin_svc.predict(X_test))
    print("Ex9 - LinearSVC test accuracy:", lin_svc_acc)

    # Cross-check against SVC(kernel="linear") and an SGDClassifier tuned to
    # approximate a linear SVM (hinge loss + matching L2 penalty) — the book
    # asks you to compare these to confirm they land in a similar ballpark.
    svc_linear = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1, random_state=random_state),
    )
    svc_linear.fit(X_train, y_train)
    svc_linear_acc = accuracy_score(y_test, svc_linear.predict(X_test))
    print("Ex9 - SVC(kernel='linear') test accuracy:", svc_linear_acc)

    sgd_clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="hinge", alpha=1 / (len(X_train) * 1),  # alpha ~ 1/(m*C)
            max_iter=10_000, tol=1e-3, random_state=random_state,
        ),
    )
    sgd_clf.fit(X_train, y_train)
    sgd_acc = accuracy_score(y_test, sgd_clf.predict(X_test))
    print("Ex9 - SGDClassifier (hinge loss) test accuracy:", sgd_acc)

    # All three should score similarly (typically 95-100% on wine, since it's
    # a small, fairly separable dataset) — that convergence is the point of
    # the exercise: LinearSVC, SVC(kernel='linear'), and SGDClassifier with
    # hinge loss are all solving approximately the same optimization problem,
    # just with different solvers/complexity trade-offs.
    return {
        "LinearSVC": lin_svc_acc,
        "SVC_linear": svc_linear_acc,
        "SGDClassifier": sgd_acc,
    }


# ---------------------------------------------------------------------------
# 10. SVM classifier on wine — tuned via RandomizedSearchCV, one-vs-rest
# ---------------------------------------------------------------------------
def exercise_10(random_state=42):
    wine = load_wine()
    X, y = wine["data"], wine["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # SVC handles multiclass internally via one-vs-one by default; the
    # exercise phrasing ("based on the wine's chemical analysis") just wants
    # a properly tuned multiclass SVM — scaling + a randomized C/gamma search.
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(random_state=random_state),
    )

    param_distribs = {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": reciprocal(0.1, 1000),
        "svc__gamma": reciprocal(0.001, 10),
    }

    rnd_search = RandomizedSearchCV(
        pipeline, param_distribs, n_iter=100, cv=5,
        scoring="accuracy", random_state=random_state, n_jobs=-1, verbose=1,
    )
    rnd_search.fit(X_train, y_train)

    print("Ex10 - Best params:", rnd_search.best_params_)
    print("Ex10 - Best CV accuracy:", rnd_search.best_score_)

    y_pred = rnd_search.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Ex10 - Test accuracy:", test_acc)
    # Expect very high accuracy here (often 97-100%) — wine's 3 cultivars
    # are well separated by the 13 chemical features once scaled.
    return rnd_search, test_acc


# ---------------------------------------------------------------------------
# 11. SVR on California housing, tuned via RandomizedSearchCV
# ---------------------------------------------------------------------------
def exercise_11(n_train_subsample=None, random_state=42):
    """
    n_train_subsample: if set, trains on a random subsample of this many
    rows (SVR doesn't scale well past a few thousand rows — the book
    explicitly recommends subsampling if you want the search to finish in
    a reasonable time).
    """
    housing = fetch_california_housing()
    X, y = housing["data"], housing["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if n_train_subsample is not None and n_train_subsample < len(X_train):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=n_train_subsample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    pipeline = make_pipeline(StandardScaler(), SVR())

    param_distribs = {
        "svr__kernel": ["linear", "rbf"],
        "svr__C": reciprocal(1, 200000),
        "svr__gamma": reciprocal(0.001, 10),
    }

    rnd_search = RandomizedSearchCV(
        pipeline, param_distribs, n_iter=50, cv=3,
        scoring="neg_mean_squared_error", random_state=random_state,
        n_jobs=-1, verbose=1,
    )
    rnd_search.fit(X_train, y_train)

    best_rmse = np.sqrt(-rnd_search.best_score_)
    print("Ex11 - Best params:", rnd_search.best_params_)
    print("Ex11 - Best CV RMSE:", best_rmse)

    y_pred = rnd_search.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Ex11 - Test RMSE:", test_rmse)
    # Note: y is in units of $100,000s in this dataset (not raw dollars), so
    # a test RMSE of e.g. 0.58 means ~$58,000 average error.
    return rnd_search, test_rmse


if __name__ == "__main__":
    print("=== Exercise 9 ===")
    exercise_9()
    print("\n=== Exercise 10 ===")
    exercise_10()
    print("\n=== Exercise 11 (subsampled to 3000 rows for speed) ===")
    exercise_11(n_train_subsample=3000)
