"""
HW2 - Problem 2: Cross-Validation

You will implement deterministic K-fold splitting (no shuffling),
and compute CV MSE for polynomial regression.

You may reuse functions from problem1.py (import allowed).
"""

from __future__ import annotations

from typing import List, Tuple, Sequence
import numpy as np

# Reuse pipeline utilities from Problem 1
from problem1 import make_poly_pipeline, mse, predict


def kfold_indices(n: int, K: int) -> List[np.ndarray]:
    """
    Deterministically split indices {0,...,n-1} into K folds in order (no shuffling).

    Return
    ------
    folds : list of length K
        folds[i] is a 1D np.ndarray of validation indices for fold i.
    """
    # TODO: Implementraise NotImplementedError
    # raise NotImplementedError

    arr = np.arange(n)

    folds = np.array_split(arr, K)

    return folds


def train_val_split(
    X: np.ndarray, y: np.ndarray, folds: List[np.ndarray], i: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use folds[i] as validation indices, and the rest as training.
    Return (Xti, yti, Xvi, yvi).
    """
    # TODO: Implement
    # raise NotImplementedError

    val_i = folds[i]

    train_i = np.concatenate(folds[:i]+folds[i+1:])

    X_train = X[train_i]
    y_train = y[train_i]

    X_val = X[val_i]
    y_val = y[val_i]
    
    return X_train, y_train, X_val, y_val


def cv_mse_poly(Xtr: np.ndarray, ytr: np.ndarray, K: int, degree: int = 3) -> float:
    """
    Perform K-fold CV on (Xtr, ytr) for polynomial regression of given degree.

    For each fold:
      - fit polynomial pipeline on training split
      - compute validation MSE
    Return the average validation MSE across folds.
    """
    # TODO: Implement
    # raise NotImplementedError
    n = len(Xtr)
    folds = kfold_indices(n,K)

    mse_list = []

    for i in range(K):
        X_train, y_train, X_val, y_val = train_val_split(Xtr, ytr, folds, i)
        pipeline = make_poly_pipeline(degree)
        pipeline.fit(X_train, y_train)

        y_pred = predict(pipeline, X_val)

        fold_mse = mse(y_pred, y_val)
        mse_list.append(fold_mse)

    return float(np.mean(mse_list))

def cv_curve(
    Xtr: np.ndarray, ytr: np.ndarray, degrees: Sequence[int], K: int = 5
) -> np.ndarray:
    """
    Return cv_mses array aligned with degrees.
    """
    # # TODO: Implement
    # raise NotImplementedError

    cv_mses = []

    for degree in degrees:
        cv_mse = cv_mse_poly(Xtr, ytr, K, degree)
        cv_mses.append(cv_mse)

    return np.array(cv_mses)



def recommend_degree_cv(degrees: Sequence[int], cv_mses: np.ndarray) -> int:
    """
    Return the degree with the smallest CV MSE.
    Break ties by returning the smaller degree.
    """
    # TODO: Implement
    # raise NotImplementedError

    best_degree = int(np.argmin(cv_mses))

    return degrees[best_degree]
