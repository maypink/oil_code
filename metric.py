import numpy as np


def wnrmse(y_true: np.array, y_pred: np.array) -> float:
    sum = 0
    for y_true_col, y_pred_col in zip(y_true.T, y_pred.T):
        res = np.sqrt(np.sum(((y_true_col - y_pred_col) / np.std(y_true_col, ddof=1)) ** 2) / y_true_col.shape[0])
        sum += res
    return sum / 5