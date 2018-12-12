import numpy as np
from sklearn.metrics import classification_report


def accuracy(clf, coeffs):
    acc = 0
    n = len(coeffs)
    lb = clf.bootstrap_CI[0]
    ub = clf.bootstrap_CI[1]
    for i, c in enumerate(coeffs):
        if ub[i] >= c >= lb[i]:
            acc += 1
    return acc / n


def coeffs_to_effects(coeffs):
    # 1 : poisitive effect
    # -1: negative effect
    # 0 : no effect
    c = np.exp(coeffs)
    effects = np.zeros_like(c)
    effects[c > 1] = 1
    effects[c < 1] = -1
    return effects


def CI_to_effects(bootstrap_CI):
    lb = np.exp(bootstrap_CI[0])
    ub = np.exp(bootstrap_CI[1])
    effects = np.zeros_like(lb)
    effects[lb > 1] = 1
    effects[ub < 1] = -1
    return effects


def model_performance(clf, coeffs):
    true = coeffs_to_effects(coeffs)
    pred = CI_to_effects(clf.bootstrap_CI)
    return classification_report(true, pred, labels=[1, 0, -1],
                                 target_names=["risk increase", "no effect",
                                               "risk decrease"], digits=3)


def squared_error(coeffs, estimates):
    # TODO check dim here
    # TODO: check types (array, list of arrays)
    theta = coeffs.ravel()
    se = np.sum(np.abs(theta - estimates.ravel())**2)
    return se


def absolute_error(coeffs, estimates):
    # TODO check dim here
    # TODO: check types (array, list of arrays)
    theta = coeffs.ravel()
    se = np.sum(np.abs(theta - estimates.ravel()))
    return se


def absolute_percentage_error(coeffs, estimates):
    # TODO check dim here
    # TODO: check types (array, list of arrays)
    theta = coeffs.ravel()
    se = np.sum(np.abs(theta - estimates.ravel())/np.abs(theta))
    return se


def mse(coeffs, estimates):
    n = len(coeffs)
    return squared_error(coeffs, estimates) / n


def mae(coeffs, estimates):
    n = len(coeffs)
    return absolute_error(coeffs, estimates) / n


def mape(coeffs, estimates):
    n = len(coeffs)
    return absolute_percentage_error(coeffs, estimates) / n
