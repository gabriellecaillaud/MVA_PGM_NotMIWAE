import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from mva_project.MVA_PGM_NotMIWAE.data_imputation import compute_rmse
from mva_project.MVA_PGM_NotMIWAE.not_miwae import notMIWAE, get_notMIWAE
from sklearn.model_selection import train_test_split
import logging

def introduce_missing(X):
    N, D = X.shape
    Xnan = X.copy()

    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz


def train_notMIWAE(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_rmse  = 0
        for batch_idx, (x, s, xtrue) in enumerate(train_loader):
            x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
            optimizer.zero_grad()
            mu, lpxz, lpmz, lqzx, lpz = model(x, s)
            loss = -get_notMIWAE(x.shape[0], lpxz, lpmz, lqzx, lpz)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            # compute rmse on batch
            batch_rmse = compute_rmse(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
            train_rmse += batch_rmse

        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)

        model.eval()
        val_loss = 0
        val_rmse = 0
        with torch.no_grad():
            for x, s, xtrue in val_loader:
                x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
                mu, lpxz, lpmz, lqzx, lpz = model(x, s)
                loss = -get_notMIWAE(x.shape[0], lpxz, lpmz, lqzx, lpz)
                val_loss += loss.item()
                batch_rmse = compute_rmse(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
                val_rmse += batch_rmse
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f} , Train rmse: {train_rmse:.4f} , Val Loss: {val_loss:.4f} , Val RMSE: {val_rmse:.4f}  last value of lr: {scheduler.get_last_lr()[-1]:.4f}')


if __name__ == "__main__":

    logging.info("Starting data preparation")
    # white wine
    data = np.array(pd.read_csv('datasets/winequality-white.csv', low_memory=False, sep=';'))

    # Split data into features and target (assuming target is the last column)
    X_data = data[:, :-1]  # Features
    y_data = data[:, -1]   # Target values (labels)

    # Split data into train and validation sets
    Xtrain, Xval, ytrain, yval = train_test_split(X_data, y_data, test_size=0.2, random_state=42, shuffle=True)

    # Compute mean and standard deviation from the training set only
    mean_train = np.mean(Xtrain, axis=0)
    std_train = np.std(Xtrain, axis=0)

    # Standardize the training set using its mean and std
    Xtrain = (Xtrain - mean_train) / std_train

    # Standardize the validation set using the training set's mean and std
    Xval = (Xval - mean_train) / std_train

    # Introduce missing data to features (only in X, not y)
    Xnan_train, Xz_train = introduce_missing(Xtrain)  # Assuming introduce_missing is defined elsewhere
    Xnan_val, Xz_val = introduce_missing(Xval)

    # ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    # yval_tensor = torch.tensor(yval, dtype=torch.float32)

    # Create missing data masks (1 if present, 0 if missing)
    Strain = torch.tensor(~np.isnan(Xnan_train), dtype=torch.float32)
    Sval = torch.tensor(~np.isnan(Xnan_val), dtype=torch.float32)

    # Convert features and target to PyTorch tensors
    Xnan_train = torch.tensor(Xnan_train, dtype=torch.float32)
    Xnan_val = torch.tensor(Xnan_val, dtype=torch.float32)
    Xtrain    = torch.tensor(Xtrain, dtype= torch.float32)
    Xval = torch.tensor(Xval, dtype= torch.float32)

    # Replace missing values (NaN) with zeros for training
    Xnan_train[torch.isnan(Xnan_train)] = 0
    Xnan_val[torch.isnan(Xnan_val)] = 0

    # Prepare TensorDatasets and DataLoaders for features (X), mask (S), and target (y)
    train_dataset = TensorDataset(Xnan_train, Strain, Xtrain) # Features, mask, true features
    val_dataset = TensorDataset(Xnan_val, Sval, Xval)



    device = "cuda" if torch.cuda.is_available() else 'cpu'


    calib_config = [{'lr': 1e-3, 'epochs' : 100, 'pct_start': 0.2, 'final_div_factor': 1e3, 'batch_size': 16, 'n_hidden': 128, 'n_latent': 10 }
                    ][-1]

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=calib_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=calib_config['batch_size'], shuffle=False)

    logging.info("Dataset prepared. Loading model")
    model = notMIWAE(n_input_features=11, n_hidden=calib_config['n_hidden'], n_latent = calib_config['n_latent'])

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=calib_config['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = calib_config['lr'],
                                                    epochs = calib_config['epochs'],
                                                    steps_per_epoch= len(train_loader),
                                                    pct_start= calib_config['pct_start'],
                                                    # final_div_factor=calib_config['final_div_factor']
                                                    )
    logging.info("Starting training")
    train_notMIWAE(model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler = scheduler, num_epochs = 100, device = device)


