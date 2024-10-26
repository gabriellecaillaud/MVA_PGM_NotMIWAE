import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import datetime

from sklearn.model_selection import train_test_split

from introduce_missing_data import  introduce_missing_mean_values,introduce_missing, introduce_missing_extreme_values
from utils import seed_everything

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(date)
    print('Run baselines')
    seed_everything(0)
    # white wine
    data = np.array(pd.read_csv('datasets/cancer-dataset/Cancer_Data.csv', low_memory=False, sep=','))

    # Split data into features and target (assuming target is the last column)
    X_data = data[:, 2:-2]  # Features
    y_data = data[:, -2]  # Target values (labels)

    # Split data into train and validation sets
    Xtrain, Xval, ytrain, yval = train_test_split(X_data, y_data, test_size=0.2, random_state=42, shuffle=True)

    # Compute mean and standard deviation from the training set only
    mean_train = np.mean(Xtrain.astype(np.float64), axis=0)
    std_train = np.std(Xtrain.astype(np.float64), axis=0)

    # Standardize the training set using its mean and std
    Xtrain = (Xtrain - mean_train) / std_train
    total_samples_x_train = Xtrain.shape[0]
    # Standardize the validation set using the training set's mean and std
    Xval = (Xval - mean_train) / std_train

    # Introduce missing data to features (only in X, not y)
    Xnan_train, Xz_train = introduce_missing_mean_values(Xtrain)  # Assuming introduce_missing is defined elsewhere
    Xnan_val, Xz_val = introduce_missing_mean_values(Xval)

    # Create missing data masks (1 if present, 0 if missing)
    Strain = np.array(~np.isnan(Xnan_train), dtype=np.float32)
    Sval = np.array(~np.isnan(Xnan_val), dtype=np.float32)

    Xtrain = Xtrain.astype(np.float32)
    Xval = Xval.astype(np.float32)

    estimator = RandomForestRegressor(n_estimators=100)
    imp = IterativeImputer(estimator=estimator)
    imp.fit(Xnan_train)
    Xrec = imp.transform(Xnan_val)
    rmse_mf = np.sqrt(np.sum((Xval - Xrec) ** 2 * (1 - Sval)) / np.sum(1 - Sval))

    print("missForst imputation RMSE: ", rmse_mf)

    imp = IterativeImputer(max_iter=100)
    imp.fit(Xnan_train)
    print("Computing results on val")
    Xrec = imp.transform(Xnan_val)
    RMSE_iter = np.sqrt(np.sum((Xval - Xrec) ** 2 * (1 - Sval)) / np.sum(1 - Sval))

    print("MICE, imputation RMSE", RMSE_iter)
