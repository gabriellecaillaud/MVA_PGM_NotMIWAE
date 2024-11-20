import datetime
from torch.utils.data import Subset
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

from data_imputation import compute_imputation_rmse_not_miwae, softmax
from not_miwae import get_notMIWAE, notMIWAE
from utils import seed_everything


class ZeroBlueTransform:
    def __init__(self, do_flatten = True):
        print("Missing data type: blue channel put to zero when blue is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where blue is the dominant color channel
        blue_dominant_mask = (img[2, :, :] > img[0, :, :]) & (img[2, :, :] > img[1, :, :])

        # Zero out pixels where blue is not dominant
        img_zero = img.clone()
        img_zero[2,blue_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[2, blue_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img

class ZeroGreenTransform:
    def __init__(self, do_flatten = True):
        print("Missing data type: GREEN channel put to zero when GREEN is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where GREEN is the dominant color channel
        blue_dominant_mask = (img[1, :, :] > img[0, :, :]) & (img[1, :, :] > img[2, :, :])

        # Zero out pixels where green is not dominant
        img_zero = img.clone()
        img_zero[1,blue_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[1, blue_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img

class ZeroRedTransform:
    def __init__(self, do_flatten = True):
        print("Missing data type: RED channel put to zero when RED is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where GREEN is the dominant color channel
        blue_dominant_mask = (img[0, :, :] > img[1, :, :]) & (img[0, :, :] > img[2, :, :])

        # Zero out pixels where green is not dominant
        img_zero = img.clone()
        img_zero[0,blue_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[0, blue_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img


class ZeroPixelWhereBlueTransform:
    def __init__(self, do_flatten = True):
        print("Type of missing data: all channels put to zero when blue is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where blue is the dominant color channel
        blue_dominant_mask = (img[2, :, :] > img[0, :, :]) & (img[2, :, :] > img[1, :, :])

        # Zero out pixels where blue is not dominant
        img_zero = img.clone()
        img_zero[:,blue_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[:, blue_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_notMIWAE_on_cifar10(model, train_loader, val_loader, optimizer, scheduler, num_epochs, total_samples_x_train, device, date):
    model.to(device)
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss  = 0
        train_rmse  = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            x, s,xtrue = data[0]
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
            for data in val_loader:
                x, s, xtrue = data[0]
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


if __name__ == "__main__":
    calib_config = [
        {'model': 'not_miwae', 'lr': 1e-3, 'epochs': 100, 'pct_start': 0.1, 'final_div_factor': 1e5, 'batch_size': 16,
         'n_hidden': 512, 'n_latent': 128, 'missing_process': 'nonlinear', 'weight_decay': 0, 'betas': (0.9, 0.999),
         'random_seed': 0, 'out_dist': 'gauss', 'dataset_size' : None, 'transform': 'ZeroBlueTransform'},
        ][-1]


    if calib_config['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform()
        ])
    elif calib_config['transform'] == 'ZeroPixelWhereBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroPixelWhereBlueTransform()
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

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(date)



    seed_everything(calib_config['random_seed'])

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = notMIWAE(n_input_features=3*32*32, n_hidden=calib_config['n_hidden'],
                     n_latent=calib_config['n_latent'], missing_process=calib_config['missing_process'],
                     out_dist=calib_config['out_dist'])

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
    train_notMIWAE_on_cifar10(model=model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, scheduler=scheduler, num_epochs = calib_config['epochs'], total_samples_x_train= 10, device=device, date=date )



