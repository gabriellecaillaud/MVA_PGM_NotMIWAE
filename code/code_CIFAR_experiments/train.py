import datetime
import os
import numpy as np
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms

from code.code_CIFAR_experiments.model import ConvNotMIWAE
from code.common.data_imputation import compute_imputation_rmse_conv_not_miwae
from code.code_UCI_experiments.not_miwae import get_notMIWAE
from code.code_CIFAR_experiments.transforms import ZeroBlueTransform, ZeroPixelWhereBlueTransform, ZeroGreenTransform, ZeroRedTransform
from code.common.utils import seed_everything


def train_conv_notMIWAE_on_cifar10(model, train_loader, val_loader, optimizer, scheduler, num_epochs,
                                   total_samples_x_train, device, date):
    os.makedirs("temp", exist_ok=True) # check if temp directory exists for saving model
    model.to(device)
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_rmse = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            x, s, xtrue = data[0]
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
                batch_rmse = compute_imputation_rmse_conv_not_miwae(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
                train_rmse += batch_rmse

        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)

        model.eval()
        val_loss = 0
        val_rmse = 0
        with torch.no_grad():
            for data in val_loader:
                x, s, xtrue = data[0]
                x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
                mu, lpxz, lpmz, lqzx, lpz = model(x, s, total_samples_x_train)
                loss = -get_notMIWAE(total_samples_x_train, lpxz, lpmz, lqzx, lpz)
                val_loss += loss.item()
                batch_rmse = compute_imputation_rmse_conv_not_miwae(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
                val_rmse += batch_rmse
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"temp/not_miwae_{date}_best_val_loss.pt")

        print(
            f'Epoch {(epoch + 1):4.0f}, Train Loss: {train_loss:8.4f} , Train rmse: {train_rmse:7.4f} , Val Loss: {val_loss:8.4f} , Val RMSE: {val_rmse:7.4f}  last value of lr: {scheduler.get_last_lr()[-1]:.4f}')


if __name__ == "__main__":

    calib_config = [
        {'model': 'not_miwae', 'lr': 1e-3, 'epochs': 100, 'pct_start': 0.1, 'final_div_factor': 1e4, 'batch_size': 8,
         'n_hidden': 512, 'n_latent': 128, 'missing_process': 'selfmasking', 'weight_decay': 0, 'betas': (0.9, 0.999),
         'random_seed': 0, 'out_dist': 'gauss', 'dataset_size' : None, 'transform': 'ZeroBlueTransform', 'hidden_dims' : [64,128,256]},
        ][-1]

    if calib_config['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
        ])
    elif calib_config['transform'] == 'ZeroGreenTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroGreenTransform(do_flatten=False)
        ])
    elif calib_config['transform'] == 'ZeroRedTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroRedTransform(do_flatten=False)
        ])
    elif calib_config['transform'] == 'ZeroPixelWhereBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroPixelWhereBlueTransform(do_flatten=False)
        ])
    else:
        raise KeyError('Transforms is not correctly defined.')
    batch_size = calib_config['batch_size']

    if calib_config['dataset_size'] is not None:
        # set download to True if this is the first time you are running this file
        train_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                download=False, transform=transform), torch.arange(calib_config['dataset_size']))

        test_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                       download=False, transform=transform), torch.arange(calib_config['dataset_size']))
    else:
        train_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                download=False, transform=transform)

        test_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                       download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(date)

    seed_everything(calib_config['random_seed'])

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = ConvNotMIWAE(n_latent=calib_config['n_latent'],
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            hidden_dims= calib_config['hidden_dims'],
            missing_process=calib_config['missing_process'])

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
    train_conv_notMIWAE_on_cifar10(model=model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, scheduler=scheduler, num_epochs = calib_config['epochs'], total_samples_x_train= 10, device=device, date=date )



