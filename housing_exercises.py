"""
Solutions to the end-of-chapter exercises (Géron, Hands-On ML, Ch.2 — housing dataset)

Assumes you already have, from earlier in the chapter:
    housing            -> raw DataFrame (features only, no target)
    housing_labels     -> Series of target values (median_house_value)
    housing_prepared   -> numpy array, output of full_pipeline.fit_transform(housing)
    full_pipeline      -> ColumnTransformer (num_pipeline + cat OneHotEncoder)
    num_attribs        -> list of numeric column names
    cat_attribs        -> list of categorical column names

If you don't have these yet, the block below reconstructs them the standard way.
Skip it if you already have housing_prepared etc. in your notebook.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# 0. RECONSTRUCT DATA + PIPELINE (skip if you already have these)
# ---------------------------------------------------------------------------
# housing = pd.read_csv("housing.csv")
# housing["income_cat"] = pd.cut(housing["median_income"],
#                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                 labels=[1, 2, 3, 4, 5])
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(housing, housing["income_cat"]):
#     strat_train_set = housing.loc[train_index]
#     strat_test_set = housing.loc[test_index]
# for set_ in (strat_train_set, strat_test_set):
#     set_.drop("income_cat", axis=1, inplace=True)
#
# housing = strat_train_set.drop("median_house_value", axis=1)
# housing_labels = strat_train_set["median_house_value"].copy()
#
# rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
#
# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def __init__(self, add_bedrooms_per_room=True):
#         self.add_bedrooms_per_room = add_bedrooms_per_room
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
#         population_per_household = X[:, population_ix] / X[:, households_ix]
#         if self.add_bedrooms_per_room:
#             bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
#             return np.c_[X, rooms_per_household, population_per_household,
#                          bedrooms_per_room]
#         return np.c_[X, rooms_per_household, population_per_household]
#
# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("attribs_adder", CombinedAttributesAdder()),
#     ("std_scaler", StandardScaler()),
# ])
#
# num_attribs = list(housing.drop("ocean_proximity", axis=1))
# cat_attribs = ["ocean_proximity"]
#
# full_pipeline = ColumnTransformer([
#     ("num", num_pipeline, num_attribs),
#     ("cat", OneHotEncoder(), cat_attribs),
# ])
#
# housing_prepared = full_pipeline.fit_transform(housing)


# ---------------------------------------------------------------------------
# 1. SVR + GridSearchCV (on first 5000 instances, 3-fold CV)
# ---------------------------------------------------------------------------
def exercise_1(housing_prepared, housing_labels):
    X_small = housing_prepared[:5000]
    y_small = housing_labels[:5000]

    param_grid = [
        {"kernel": ["linear"], "C": [10., 30., 100., 300., 1000., 3000.]},
        {"kernel": ["rbf"], "C": [1.0, 3.0, 10., 30., 100., 300., 1000.],
         "gamma": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

    svm_reg = SVR()
    grid_search = GridSearchCV(svm_reg, param_grid, cv=3,
                                scoring="neg_mean_squared_error",
                                verbose=2, n_jobs=-1)
    grid_search.fit(X_small, y_small)

    best_rmse = np.sqrt(-grid_search.best_score_)
    print("Ex1 - Best params:", grid_search.best_params_)
    print("Ex1 - Best CV RMSE:", best_rmse)
    # In the book's run, the rbf kernel wins but RMSE (~70k) is worse than the
    # RandomForestRegressor from earlier in the chapter (~50k) — SVR needs
    # more careful tuning / feature scaling / more data to compete.
    return grid_search


# ---------------------------------------------------------------------------
# 2. RandomizedSearchCV instead of GridSearchCV
# ---------------------------------------------------------------------------
def exercise_2(housing_prepared, housing_labels):
    X_small = housing_prepared[:5000]
    y_small = housing_labels[:5000]

    param_distribs = {
        "kernel": ["linear", "rbf"],
        "C": stats.reciprocal(20, 200000),
        "gamma": stats.expon(scale=1.0),
    }

    svm_reg = SVR()
    rnd_search = RandomizedSearchCV(
        svm_reg, param_distributions=param_distribs,
        n_iter=50, cv=3, scoring="neg_mean_squared_error",
        verbose=2, random_state=42, n_jobs=-1,
    )
    rnd_search.fit(X_small, y_small)

    best_rmse = np.sqrt(-rnd_search.best_score_)
    print("Ex2 - Best params:", rnd_search.best_params_)
    print("Ex2 - Best CV RMSE:", best_rmse)
    # reciprocal/expon distributions concentrate search on the
    # order-of-magnitude ranges that mattered in exercise 1, so it usually
    # finds a comparable or better result in far fewer fits.
    return rnd_search


# ---------------------------------------------------------------------------
# 3. SelectFromModel in the preparation pipeline
# ---------------------------------------------------------------------------
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """Keeps only the k most important features, given precomputed importances."""
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]


def exercise_3(full_pipeline, housing, housing_prepared, housing_labels, k=5):
    # Fit a RandomForest just to get feature importances (any model with
    # feature_importances_ or coef_ works with SelectFromModel too).
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    feature_importances = forest_reg.feature_importances_

    # Option A: sklearn's built-in SelectFromModel
    selector = SelectFromModel(forest_reg, threshold=-np.inf,
                                max_features=k, prefit=True)
    housing_top_k_a = selector.transform(housing_prepared)

    # Option B: custom transformer (matches the book's approach, useful when
    # you want to plug it directly into a Pipeline with GridSearchCV over k)
    prep_and_select_pipeline = Pipeline([
        ("preparation", full_pipeline),
        ("feature_selection", TopFeatureSelector(feature_importances, k)),
    ])
    housing_top_k_b = prep_and_select_pipeline.fit_transform(housing)

    print("Ex3 - top-k feature indices:",
          indices_of_top_k(feature_importances, k))
    print("Ex3 - shapes:", housing_top_k_a.shape, housing_top_k_b.shape)
    return prep_and_select_pipeline, housing_top_k_b


# ---------------------------------------------------------------------------
# 4. Custom KNN-based transformer adding a "nearest-district price" feature
# ---------------------------------------------------------------------------
class KNNPriceAdder(BaseEstimator, TransformerMixin):
    """
    Trains a KNeighborsRegressor on (lat, lon) -> median_house_value in fit(),
    then in transform() appends the predicted price for each row's location
    as a new feature.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        # X here is the raw housing DataFrame (or array) containing at least
        # 'latitude' and 'longitude'; y must be the price labels.
        coords = self._get_coords(X)
        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.knn_.fit(coords, y)
        return self

    def transform(self, X):
        coords = self._get_coords(X)
        predicted_price = self.knn_.predict(coords).reshape(-1, 1)
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.c_[X_arr, predicted_price]

    @staticmethod
    def _get_coords(X):
        if isinstance(X, pd.DataFrame):
            return X[["latitude", "longitude"]].values
        # assumes latitude/longitude are columns 0 and 1 if a raw array
        return np.asarray(X)[:, :2]


def exercise_4(housing, housing_labels):
    knn_adder = KNNPriceAdder(n_neighbors=5)
    housing_with_knn_feature = knn_adder.fit_transform(housing, housing_labels)
    print("Ex4 - new shape after adding KNN price feature:",
          housing_with_knn_feature.shape)

    # To slot this into the full preprocessing pipeline (so it runs before
    # scaling/imputing), you'd typically add it as another branch in a
    # ColumnTransformer/FeatureUnion alongside num_pipeline and cat_pipeline,
    # since it needs raw lat/lon (and y) rather than the already-scaled output.
    return knn_adder, housing_with_knn_feature


# ---------------------------------------------------------------------------
# 5. GridSearchCV over the whole preparation pipeline
# ---------------------------------------------------------------------------
def exercise_5(full_pipeline, forest_reg_feature_importances, housing,
               housing_labels):
    full_prep_and_predict_pipeline = Pipeline([
        ("preparation", full_pipeline),
        ("feature_selection", TopFeatureSelector(forest_reg_feature_importances, k=5)),
        ("svm_reg", SVR()),
    ])

    param_grid = [{
        "preparation__num__attribs_adder__add_bedrooms_per_room": [False, True],
        "feature_selection__k": [3, 5, 7, 9],
        "svm_reg__kernel": ["linear", "rbf"],
        "svm_reg__C": [10., 100., 1000.],
    }]

    grid_search_prep = GridSearchCV(
        full_prep_and_predict_pipeline, param_grid, cv=3,
        scoring="neg_mean_squared_error", verbose=2, n_jobs=-1,
    )
    grid_search_prep.fit(housing[:5000], housing_labels[:5000])

    print("Ex5 - best params:", grid_search_prep.best_params_)
    print("Ex5 - best CV RMSE:", np.sqrt(-grid_search_prep.best_score_))
    return grid_search_prep


# ---------------------------------------------------------------------------
# 6. StandardScalerClone from scratch, with inverse_transform + DataFrame support
# ---------------------------------------------------------------------------
from sklearn.utils.validation import check_array, check_is_fitted


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        # capture column names BEFORE check_array strips them to a bare ndarray
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        X = check_array(X)  # validates + converts to numpy array
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Unexpected number of features")
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Unexpected number of features")
        X = X * self.scale_
        return X + self.mean_ if self.with_mean else X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError("Invalid number of features")
            if hasattr(self, "feature_names_in_") and not np.array_equal(
                self.feature_names_in_, input_features
            ):
                raise ValueError("input_features != feature_names_in_")
            return np.asarray(input_features, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        else:
            return np.array([f"x{i}" for i in range(self.n_features_in_)],
                             dtype=object)


def exercise_6_demo():
    from sklearn.utils.estimator_checks import check_estimator
    X = np.random.rand(100, 3)
    scaler = StandardScalerClone()
    X_scaled = scaler.fit_transform(X)
    X_back = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_back)

    # DataFrame input -> feature names remembered
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    scaler2 = StandardScalerClone().fit(df)
    print("Ex6 - feature names in:", scaler2.get_feature_names_out())

    # Sanity check against sklearn's own scaler (values should match)
    sk_scaler = StandardScaler().fit(X)
    assert np.allclose(sk_scaler.transform(X), scaler.transform(X))
    print("Ex6 - matches sklearn StandardScaler: OK")
    return scaler


if __name__ == "__main__":
    print("This module defines functions/classes for each exercise.")
    print("Import housing_prepared/housing_labels/full_pipeline from your")
    print("notebook, then call e.g. exercise_1(housing_prepared, housing_labels).")
    exercise_6_demo()
