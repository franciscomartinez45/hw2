"""
HW2 - Problem 1: Linear/Polynomial Regression + Model Selection

You may use: numpy, pandas, scikit-learn.
Do NOT use any course-bundled / legacy ML libraries.

All functions must keep the same names/signatures.
The autograder will import and call these functions directly.
"""

from __future__ import annotations

from typing import Tuple, Sequence, Dict, Any
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler




def load_curve80(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load curve80.txt where column 0 is x and column 1 is y.

    Returns
    -------
    X : np.ndarray of shape (N, 1)
    y : np.ndarray of shape (N,)
    """
    # # TODO: Implement loading
    # raise NotImplementedError
    data = np.loadtxt(fname=path)
    X = data[:,0].reshape(-1,1)
    y = data[:,1]


    return X,y



def split_data(
    X: np.ndarray, y: np.ndarray, frac: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministically split without shuffling:
    first frac portion for train, remaining for test.
    """
    # TODO: Implement deterministic split
    # raise NotImplementedError
    split = int(X.shape[0]*frac)

    return X[:split], X[split:], y[:split], y[split:]



def shapes(
    Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Return shapes as tuples: (Xtr.shape, Xte.shape, ytr.shape, yte.shape)
    """
    # TODO: Implement
    # raise NotImplementedError
    return Xtr.shape, Xte.shape, ytr.shape, yte.shape

def fit_linear(Xtr: np.ndarray, ytr: np.ndarray) -> LinearRegression:
    """
    Fit a baseline linear regression model with intercept.
    """
    # TODO: Implement training
    # raise NotImplementedError
    model = LinearRegression(fit_intercept=True)
    model.fit(Xtr, ytr)
    return model

def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return predictions as a 1D array of length N.
    """
    # TODO: Implement prediction wrapper
    # raise NotImplementedError

    y_predict = model.predict(X)
    return y_predict.flatten()

def mse(yhat: np.ndarray, y: np.ndarray) -> float:
    """
    Mean squared error.
    """
    # TODO: Implement MSE
    # raise NotImplementedError

    return float(np.mean((yhat - y)**2))


def eval_linear(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray
) -> Tuple[float, float]:
    """
    Train baseline linear regression, return (train_mse, test_mse).
    """
    # TODO: Implement evaluation
    # raise NotImplementedError

    model = fit_linear(Xtr, ytr)
    y_train_prediction = predict(model, Xtr)
    y_test_prediction = predict(model, Xte)
    train_mse = mse(y_train_prediction, ytr)
    test_mse =  mse(y_test_prediction, yte)
    return train_mse,test_mse

def make_poly_pipeline(degree: int = 3) -> Pipeline:
    """
    Create a scikit-learn pipeline:
      PolynomialFeatures(degree, include_bias=False)
      StandardScaler()
      LinearRegression(fit_intercept=True)

    Returns an *unfitted* Pipeline.
    """
    # TODO: Create and return pipeline
    # raise NotImplementedError
    steps = [('polynomial', PolynomialFeatures(degree=degree, include_bias=False)),
             ('scalar', StandardScaler()),
             ('linear'), LinearRegression(fit_intercept=True)
             ]
    return Pipeline(steps)

def fit_poly(Xtr: np.ndarray, ytr: np.ndarray, degree: int) -> Pipeline:
    """
    Fit the polynomial pipeline of the given degree on training data.
    """
    # TODO: Fit and return pipeline
    # raise NotImplementedError
    pipeline = make_poly_pipeline(degree)
    pipeline.fit(Xtr, ytr)
    return pipeline

def eval_poly(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, degree: int = 3
) -> Tuple[float, float]:
    """
    Fit polynomial regression of given degree and return (train_mse, test_mse).
    """
    # # TODO: Implement
    # raise NotImplementedError

    model = fit_poly(Xtr, ytr, degree)

    y_train_pred = predict(model,Xtr)
    y_test_pred = predict(model, Xte)

    train_mse = mse(y_train_pred, ytr)
    test_mse = mse(y_test_pred, yte)

    return train_mse, test_mse


def eval_degrees(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    degrees: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a list/tuple of degrees.
    Returns two arrays:
      mse_tr : np.ndarray shape (len(degrees),)
      mse_te : np.ndarray shape (len(degrees),)
    aligned with degrees.
    """
    # TODO: Implement
    # raise NotImplementedError
    train_errors = []
    test_errors = []

    for degree in degrees:
        train_mse, test_mse = eval_poly(Xtr, ytr, Xte, yte, degree)
        train_errors.append(train_mse)
        test_errors.append(test_mse)

    mse_tr = np.array(train_errors)
    mse_te = np.array(test_errors)

    return mse_tr, mse_te

def recommend_degree(degrees: Sequence[int], mse_te: np.ndarray) -> int:
    """
    Return the degree with the smallest test MSE.
    Break ties by returning the smaller degree.
    """
    # # TODO: Implement
    # raise NotImplementedError

    smallest_index = np.argmin(mse_te)

    return degrees[int(smallest_index)]


