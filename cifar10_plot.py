import datetime
from torch.utils.data import Subset
import torch
import torchvision
import torchvision.transforms as transforms
from convolutional_not_miwae import ConvNotMIWAE
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from data_imputation import compute_imputation_rmse_not_miwae, softmax
from not_miwae import get_notMIWAE, notMIWAE
from not_miwae_cifar import ZeroBlueTransform, ZeroPixelWhereBlueTransform
from utils import seed_everything



def plot_images(transform_name):
    if transform_name == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
        ])
    elif transform_name == 'ZeroPixelWhereBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroPixelWhereBlueTransform(do_flatten=False)
        ])
    else:
        raise KeyError('Transforms is not correctly defined.')
    train_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                    download=False, transform=transform),
                       torch.arange(10))
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                               shuffle=True, num_workers=2)

    data = next(iter(dataloader))
    img_zero_batch, img_mask_batch, original_batch = data[0]
    # Plot the images
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))

    for i in range(4):
        # Original image
        axes[i, 0].imshow(original_batch[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Transformed image with blue pixels zeroed out
        axes[i, 1].imshow(img_zero_batch[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Zero Blue Pixels")
        axes[i, 1].axis("off")

        # Mask showing unchanged and modified pixels
        axes[i, 2].imshow(img_mask_batch[i].permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 2].set_title("Modification Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

def plot_images_with_imputation(model_path, is_conv_model, calib_config):

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if calib_config['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
        ])
    elif calib_config['transform'] == 'ZeroPixelWhereBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroPixelWhereBlueTransform(do_flatten=False)
        ])
    else:
        raise KeyError('Transforms is not correctly defined.')
    val_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                    download=False, transform=transform),
                       torch.arange(4))
    dataloader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                               shuffle=True, num_workers=2)
    if is_conv_model:
            model = ConvNotMIWAE(n_latent=calib_config['n_latent'],
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            hidden_dims= calib_config['hidden_dims'],
            missing_process=calib_config['missing_process'])
    else:
        model = notMIWAE(n_input_features=3*32*32, n_hidden=calib_config['n_hidden'],
                        n_latent=calib_config['n_latent'], missing_process=calib_config['missing_process'],
                        out_dist=calib_config['out_dist'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    data = next(iter(dataloader))
    img_zero_batch, img_mask_batch, original_batch = data[0]
    with torch.no_grad():
        if is_conv_model: 
            mu, lpxz, lpmz, lqzx, lpz = model(img_zero_batch, img_mask_batch, 10)
            wl = softmax(lpxz + lpmz + lpz - lqzx)
            wl = wl.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # Compute the missing data imputation
            Xm = torch.sum(mu * wl, dim=1)
            X_imputed = img_zero_batch + Xm * (1 - img_mask_batch)
        else:
            mu, lpxz, lpmz, lqzx, lpz = model(img_zero_batch.flatten(start_dim=1), img_mask_batch.flatten(start_dim=1), 10)
            mu = torch.exp(mu)
            # Compute the importance weights
            wl = softmax(lpxz + lpmz + lpz - lqzx)
            # Compute the missing data imputation
            Xm = torch.sum((mu.T * wl.T).T, dim=0)
            X_imputed = img_zero_batch + Xm.reshape((4,3,32,32)) * (1 - img_mask_batch)

    # Plot the images
    fig, axes = plt.subplots(4, 4, figsize=(10, 12))

    for i in range(4):
        # Original image
        axes[i, 0].imshow(original_batch[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Transformed image with blue pixels zeroed out
        axes[i, 1].imshow(img_zero_batch[i].permute(1, 2, 0).numpy())  
        axes[i, 1].set_title("With missing data", wrap=True)
        axes[i, 1].axis("off")

        # Mask showing unchanged and modified pixels
        axes[i, 2].imshow(img_mask_batch[i].permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 2].set_title("Modification Mask")
        axes[i, 2].axis("off")

        # Mask showing unchanged and modified pixels
        axes[i, 3].imshow(X_imputed[i].permute(1, 2, 0).numpy())
        axes[i, 3].set_title("After imputation")
        axes[i, 3].axis("off")
    
    if calib_config['transform'] == 'ZeroPixelWhereBlueTransform':
        plt.suptitle("Pixels RGB where most blue removed", wrap=True)
    elif calib_config['transform'] == 'ZeroBlueTransform':
        plt.suptitle("Only blue pixels where most blue removed", wrap=True)

    plt.tight_layout()
    plt.savefig(f"temp/plot_{Path(model_path).stem}_res_{date}.png")

if __name__=="__main__":
    # plot_images(transform_name='ZeroBlueTransform')
    model_path = "/raid/home/detectionfeuxdeforet/caillaud_gab/mva_pgm/MVA_PGM_NotMIWAE/temp/not_miwae_2024_11_02_16_26_30_best_val_loss.pt"
    calib_config = {'n_hidden' : 512, 'n_latent': 128, 'missing_process': 'selfmasking', 'out_dist': 'gauss', 'transform': 'ZeroBlueTransform', 'hidden_dims' : [64,128,256]}
    plot_images_with_imputation(model_path, is_conv_model = True, calib_config=calib_config)
