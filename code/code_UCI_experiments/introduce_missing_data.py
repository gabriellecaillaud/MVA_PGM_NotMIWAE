import numpy as np


def introduce_missing_superior_to_mean(X):
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


def introduce_missing_random(X, percentage_to_remove=30):
    """
    Introduce missing data by randomly removing values from the input matrix

    Args:
        X: Input numpy array of shape (N, D)
        percentage_to_remove: Percentage of values to remove (default: 30)

    Returns:
        tuple: (Array with NaN values, Array with zeros instead of NaN)
    """
    print("Introducing random missing data")
    N, D = X.shape
    Xnan = X.copy()

    # Calculate total number of elements to remove
    total_elements = N * D
    num_to_remove = int(total_elements * percentage_to_remove / 100)

    # Create a flat mask of indices
    flat_indices = np.arange(total_elements)

    # Randomly select indices to remove
    indices_to_remove = np.random.choice(
        flat_indices,
        size=num_to_remove,
        replace=False
    )

    # Convert flat indices to 2D indices
    rows = indices_to_remove // D
    cols = indices_to_remove % D

    # Set selected values to NaN
    Xnan[rows, cols] = np.nan
    Xnan = Xnan.astype(np.float32)

    # Create version with zeros instead of NaN
    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    # Print some statistics
    missing_count = np.isnan(Xnan).sum()
    actual_percentage = (missing_count / total_elements) * 100
    print(f"Removed {missing_count} values ({actual_percentage:.2f}%)")

    return Xnan, Xz