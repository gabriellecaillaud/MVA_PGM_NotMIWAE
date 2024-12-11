import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import datetime
from code.common.data_imputation import compute_imputation_rmse_miwae,compute_imputation_rmse_not_miwae
from miwae import get_MIWAE, Miwae
from not_miwae import notMIWAE, get_notMIWAE
from sklearn.model_selection import train_test_split
import logging
from introduce_missing_data import introduce_missing_superior_to_mean
from code.common.utils import seed_everything


def train_notMIWAE(model, train_loader, val_loader, optimizer, scheduler, num_epochs, total_samples_x_train, device):
    model.to(device)
    os.makedirs("temp", exist_ok=True)
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss  = 0
        train_rmse  = 0
        for batch_idx, (x, s, xtrue) in enumerate(train_loader):
            model.train()
            x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
            optimizer.zero_grad()
            mu, lpxz, lpmz, lqzx, lpz = model(x, s, total_samples_x_train)
            loss = -get_notMIWAE(total_samples_x_train, lpxz, lpmz, lqzx, lpz)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            model.eval()
            # compute rmse on batch
            with torch.no_grad():
                batch_rmse = compute_imputation_rmse_not_miwae(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
                train_rmse += batch_rmse

        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)

        model.eval()
        val_loss = 0
        val_rmse = 0
        with torch.no_grad():
            for x, s, xtrue in val_loader:
                x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
                mu, lpxz, lpmz, lqzx, lpz   = model(x, s, total_samples_x_train)
                loss            = -get_notMIWAE(total_samples_x_train, lpxz, lpmz, lqzx, lpz)
                val_loss        += loss.item()
                batch_rmse      = compute_imputation_rmse_not_miwae(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
                val_rmse        += batch_rmse
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"temp/not_miwae_{date}_best_val_loss.pt")

        print(f'Epoch {(epoch + 1):4.0f}, Train Loss: {train_loss:8.4f} , Train rmse: {train_rmse:7.4f} , Val Loss: {val_loss:8.4f} , Val RMSE: {val_rmse:7.4f}  last value of lr: {scheduler.get_last_lr()[-1]:.4f}')


def train_MIWAE(model, train_loader, val_loader, optimizer, scheduler, num_epochs, total_samples_x_train, device):
    model.to(device)
    print("Traning MIWAE model and not NOT-MIWAE.")
    print("This will use the get_MIWAE loss")
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss  = 0
        train_rmse  = 0
        for batch_idx, (x, s, xtrue) in enumerate(train_loader):
            model.train()
            x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
            optimizer.zero_grad()
            mu, lpxz, lqzx, lpz = model(x, s, total_samples_x_train)
            loss = -get_MIWAE(total_samples_x_train, lpxz, lqzx, lpz)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                # compute rmse on batch
                batch_rmse = compute_imputation_rmse_miwae(mu, lpxz, lqzx, lpz, xtrue, s)
                train_rmse += batch_rmse

        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)

        model.eval()
        val_loss = 0
        val_rmse = 0
        with torch.no_grad():
            for x, s, xtrue in val_loader:
                x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
                mu, lpxz, lqzx, lpz   = model(x, s, total_samples_x_train)
                loss            = -get_MIWAE(total_samples_x_train, lpxz, lqzx, lpz)
                val_loss        += loss.item()
                batch_rmse      = compute_imputation_rmse_miwae(mu, lpxz, lqzx, lpz, xtrue, s)
                val_rmse        += batch_rmse
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"temp/miwae_{date}_best_val_loss.pt")

        print(f'Epoch {(epoch + 1):4.0f}, Train Loss: {train_loss:8.4f} , Train rmse: {train_rmse:7.4f} , Val Loss: {val_loss:8.4f} , Val RMSE: {val_rmse:7.4f}  last value of lr: {scheduler.get_last_lr()[-1]:.4f}')


if __name__ == "__main__":

    logging.info("Starting data preparation")
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(date)

    calib_config = [{'model': 'not_miwae', 'dataset_name':  'cancer', 'lr': 5e-4, 'epochs' : 500, 'pct_start': 0.2, 'final_div_factor': 1e4, 'batch_size': 32, 'n_hidden': 128, 'n_latent': 10, 'missing_process':'nonlinear', 'weight_decay': 0, 'betas': (0.9, 0.999), 'random_seed': 0, 'out_dist': 'gauss'},
                    {'model': 'not_miwae','dataset_name':  'cancer', 'lr': 5e-4, 'epochs' : 500, 'pct_start': 0.2, 'final_div_factor': 1e4, 'batch_size': 32, 'n_hidden': 128, 'n_latent': 10, 'missing_process':'selfmasking_known', 'weight_decay': 0, 'betas': (0.9, 0.999), 'random_seed': 0, 'out_dist': 't'},
                    {'model': 'not_miwae', 'dataset_name':  'white_wine', 'lr': 5e-4, 'epochs' : 500, 'pct_start': 0.2, 'final_div_factor': 1e4, 'batch_size': 32, 'n_hidden': 128, 'n_latent': 10, 'missing_process':'linear', 'weight_decay': 0, 'betas': (0.9, 0.999), 'random_seed': 0, 'out_dist': 'gauss'},
                    {'model': 'miwae', 'dataset_name':  'white_wine', 'lr': 5e-4, 'epochs' : 500, 'pct_start': 0.2, 'final_div_factor': 1e4, 'batch_size': 32, 'n_hidden': 128, 'n_latent': 10, 'missing_process':'selfmasking', 'weight_decay': 0, 'betas': (0.9, 0.999), 'random_seed': 0, 'out_dist': 'gauss'},
                    {'model': 'not_miwae', 'dataset_name':  'cancer', 'lr': 1e-4, 'epochs' : 2, 'pct_start': 0.2, 'final_div_factor': 1e4, 'batch_size': 16, 'n_hidden': 128, 'n_latent': 28, 'missing_process':'linear', 'weight_decay': 0, 'betas': (0.9, 0.999), 'random_seed': 0, 'out_dist': 'gauss'},

                    ][-1]

    seed_everything(calib_config['random_seed'])

    if calib_config['dataset_name'] == 'cancer':
        data = np.array(pd.read_csv('../../datasets/cancer-dataset/Cancer_Data.csv', low_memory=False, sep=','))
        X_data = data[:, 2:-2]  # Features

    elif calib_config['dataset_name'] == 'white_wine':
        data = np.array(pd.read_csv('../../datasets/winequality-white.csv', low_memory=False, sep=';'))
        X_data = data[:, :-1]  # Features. Not using the last colomn since it's the target

    else:
        raise ValueError("Please provide a dataset name from ['cancer', 'white_wine']")
    print(f"Number of features: {X_data.shape[1]}")
    # Split data into train and validation sets
    Xtrain, Xval = train_test_split(X_data, test_size=0.2, random_state=42, shuffle=True)

    # Compute mean and standard deviation from the training set only
    mean_train  = np.mean(Xtrain.astype(np.float64), axis=0)
    std_train   = np.std(Xtrain.astype(np.float64), axis=0)

    # Standardize the training set using its mean and std
    Xtrain      = (Xtrain - mean_train) / std_train
    total_samples_x_train = Xtrain.shape[0]
    # Standardize the validation set using the training set's mean and std
    Xval = (Xval - mean_train) / std_train

    # Introduce missing data to features
    Xnan_train, Xz_train    = introduce_missing_superior_to_mean(Xtrain)
    Xnan_val, Xz_val        = introduce_missing_superior_to_mean(Xval)

    # Create missing data masks (1 if present, 0 if missing)
    Strain  = torch.tensor(~np.isnan(Xnan_train), dtype=torch.float32)
    Sval    = torch.tensor(~np.isnan(Xnan_val), dtype=torch.float32)

    Xtrain  = Xtrain.astype(np.float32)
    Xval    = Xval.astype(np.float32)

    # Convert features and target to PyTorch tensors
    Xnan_train  = torch.tensor(Xnan_train, dtype=torch.float32)
    Xnan_val    = torch.tensor(Xnan_val, dtype=torch.float32)
    Xtrain      = torch.tensor(Xtrain, dtype= torch.float32)
    Xval        = torch.tensor(Xval, dtype= torch.float32)

    # Replace missing values (NaN) with zeros for training
    Xnan_train[torch.isnan(Xnan_train)]     = 0
    Xnan_val[torch.isnan(Xnan_val)]         = 0

    # Prepare TensorDatasets and DataLoaders for features (X), mask (S), and target (y)
    train_dataset   = TensorDataset(Xnan_train, Strain, Xtrain) # Features, mask, true features
    val_dataset     = TensorDataset(Xnan_val, Sval, Xval)

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Create DataLoaders
    train_loader    = DataLoader(train_dataset, batch_size=calib_config['batch_size'], shuffle=True)
    val_loader      = DataLoader(val_dataset, batch_size=calib_config['batch_size'], shuffle=False)

    logging.info("Dataset prepared. Loading model")

    if calib_config['model'] == 'not_miwae':
        model = notMIWAE(n_input_features=Xtrain.shape[1], n_hidden=calib_config['n_hidden'], n_latent = calib_config['n_latent'], missing_process = calib_config['missing_process'], out_dist=calib_config['out_dist'])
    elif calib_config['model'] == 'miwae':
        model = Miwae(n_input_features=Xtrain.shape[1], n_hidden=calib_config['n_hidden'], n_latent = calib_config['n_latent'], out_dist=calib_config['out_dist'])
    else:
        raise ValueError('Name of the model to train incorrect.')
    model.to(device)
    print(f"Number of parameters in the model: {sum (p.numel() if p.requires_grad else 0 for p in model.parameters()) }")
    optimizer = torch.optim.Adam(model.parameters(), lr=calib_config['lr'], weight_decay=calib_config['weight_decay'], betas=calib_config['betas'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = calib_config['lr'],
                                                    epochs = calib_config['epochs'],
                                                    steps_per_epoch= len(train_loader),
                                                    pct_start= calib_config['pct_start'],
                                                    # final_div_factor=calib_config['final_div_factor']
                                                    )
    print(f"calib_config:{calib_config}")
    logging.info("Starting training")
    if calib_config['model'] == 'not_miwae':
        train_notMIWAE(model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler = scheduler, num_epochs = calib_config['epochs'], total_samples_x_train=total_samples_x_train, device = device)
        torch.save(model.state_dict(), f"temp/not_miwae_{date}_last_epoch.pt")
    elif calib_config['model'] == 'miwae':
        train_MIWAE(model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler = scheduler, num_epochs = calib_config['epochs'], total_samples_x_train=total_samples_x_train, device = device)
        torch.save(model.state_dict(), f"temp/miwae_{date}_last_epoch.pt")


