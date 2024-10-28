import numpy as np


def introduce_missing(X):
    print("Introducing missing data with > mean")
    N, D = X.shape
    Xnan = X.copy()

    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xnan = Xnan.astype(np.float32)
    Xz = Xnan.copy()
    Xz[np.isnan(Xnan.astype(np.float32))] = 0

    return Xnan, Xz

def introduce_missing_extreme_values(X, percentile_extreme = 25):
    print("Introducing missing data via removing extreme values")
    N, D = X.shape
    Xnan = X.copy()

    # ---- MNAR in D/2 dimensions
    lower_bound = np.percentile(Xnan[:, :int(D / 2)], percentile_extreme, axis=0)
    upper_bound = np.percentile(Xnan[:, :int(D / 2)], 100 - percentile_extreme, axis=0)

    ix_lower = Xnan[:, :int(D / 2)] < lower_bound
    ix_higher = Xnan[:, :int(D / 2)] > upper_bound
    Xnan[:, :int(D / 2)][ix_lower | ix_higher] = np.nan
    Xnan = Xnan.astype(np.float32)
    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz

def introduce_missing_mean_values(X, percentage_to_remove = 30):
    print("Introduce missing data by removing the values around the mean")
    N, D = X.shape
    Xnan = X.copy()

    num_elements = int(N * percentage_to_remove / 100)   # number of elements to remove

    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    abs_diff_from_mean = np.abs(Xnan[:, :int(D / 2)] - mean)
    indices_to_remove = np.argsort(abs_diff_from_mean, axis = 0)[:num_elements]
    # Set those values to NaN
    for d in range(indices_to_remove.shape[1]):
        Xnan[indices_to_remove[:, d], d] = np.nan
    Xnan = Xnan.astype(np.float32)
    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0
    return Xnan, Xz