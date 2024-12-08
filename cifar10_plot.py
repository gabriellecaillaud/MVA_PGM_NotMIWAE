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
from not_miwae_cifar import ZeroBlueTransform, ZeroRedTransform,  ZeroPixelWhereBlueTransform, ZeroGreenTransform
from utils import seed_everything
from cifar10_baselines import mean_imputation_and_rmse



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

def plot_images_with_imputation(model_path, is_conv_model, calib_config, number_of_images=4):

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if calib_config['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
        ])
         
    elif calib_config['transform'] == 'ZeroRedTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroRedTransform(do_flatten=False)
        ])
    elif calib_config['transform'] == 'ZeroGreenTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroGreenTransform(do_flatten=False)
        ])
    elif calib_config['transform'] == 'ZeroPixelWhereBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroPixelWhereBlueTransform(do_flatten=False)
        ])
    else:
        raise KeyError('Transforms is not correctly defined.')
    val_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                    download=False, transform=transform)                       
    indices = [i for i, label in enumerate(val_set.targets) if label in [0,1,8,9]][:number_of_images]

    val_set = Subset(val_set, indices)
    seed_everything(2)
    dataloader = torch.utils.data.DataLoader(val_set, batch_size=number_of_images,
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
            X_imputed = img_zero_batch + Xm.reshape((number_of_images,3,32,32)) * (1 - img_mask_batch)

    # Plot the images
    fig, axes = plt.subplots(number_of_images, 4, figsize=(10, 12))

    for i in range(number_of_images):
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
    elif calib_config['transform'] == 'ZeroRdTransform':
        plt.suptitle("Only red pixels where most red removed", wrap=True)
    elif calib_config['transform'] == 'ZeroBlueTransform':
        plt.suptitle("Only blue pixels where most blue removed", wrap=True)
    elif calib_config['transform'] == 'ZeroGreenTransform':
        plt.suptitle("Only green pixels where most green removed", wrap=True)

    plt.tight_layout()
    plt.savefig(f"temp/plot_{Path(model_path).stem}_res_{date}.png")

def plot_images_with_imputation_comparison_between_models(model_path1, is_conv_model1, calib_config1, model_path2, is_conv_model2, calib_config2, number_of_images=4):

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print("Warning : Introducing missing data as defined in calib_config1")
    if calib_config1['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
        ])
         
    elif calib_config1['transform'] == 'ZeroRedTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroRedTransform(do_flatten=False)
        ])
    elif calib_config1['transform'] == 'ZeroGreenTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroGreenTransform(do_flatten=False)
        ])
    elif calib_config1['transform'] == 'ZeroPixelWhereBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroPixelWhereBlueTransform(do_flatten=False)
        ])
    else:
        raise KeyError('Transforms is not correctly defined.')
    val_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                    download=False, transform=transform)
    frog_indices = [i for i, label in enumerate(val_dataset.targets) if label == 6][:number_of_images]

    val_set = Subset(val_dataset,torch.arange(number_of_images))
    seed_everything(2)
    dataloader = torch.utils.data.DataLoader(val_set, batch_size=number_of_images,
                                               shuffle=True, num_workers=2)
    if is_conv_model1:
            model1 = ConvNotMIWAE(n_latent=calib_config1['n_latent'],
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            hidden_dims= calib_config1['hidden_dims'],
            missing_process=calib_config1['missing_process'])
    else:
        model1 = notMIWAE(n_input_features=3*32*32, n_hidden=calib_config1['n_hidden'],
                        n_latent=calib_config1['n_latent'], missing_process=calib_config1['missing_process'],
                        out_dist=calib_config1['out_dist'])
    model1.load_state_dict(torch.load(model_path1, weights_only=True))
    model1.eval()
    if is_conv_model2:
            model2 = ConvNotMIWAE(n_latent=calib_config2['n_latent'],
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            hidden_dims= calib_config2['hidden_dims'],
            missing_process=calib_config2['missing_process'])
    else:
        model2 = notMIWAE(n_input_features=3*32*32, n_hidden=calib_config2['n_hidden'],
                        n_latent=calib_config2['n_latent'], missing_process=calib_config2['missing_process'],
                        out_dist=calib_config2['out_dist'])
    model2.load_state_dict(torch.load(model_path2, weights_only=True))
    model2.eval()
    data = next(iter(dataloader))
    img_zero_batch, img_mask_batch, original_batch = data[0]
    img_zero_batch_copy = img_zero_batch.detach().clone()
    with torch.no_grad():
        if is_conv_model1: 
            mu, lpxz, lpmz, lqzx, lpz = model1(img_zero_batch, img_mask_batch, 10)
            wl = softmax(lpxz + lpmz + lpz - lqzx)
            wl = wl.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # Compute the missing data imputation
            Xm = torch.sum(mu * wl, dim=1)
            X_imputed_model1 = img_zero_batch + Xm * (1 - img_mask_batch)
        else:
            mu, lpxz, lpmz, lqzx, lpz = model1(img_zero_batch.flatten(start_dim=1), img_mask_batch.flatten(start_dim=1), 10)
            mu = torch.exp(mu)
            # Compute the importance weights
            wl = softmax(lpxz + lpmz + lpz - lqzx)
            # Compute the missing data imputation
            Xm = torch.sum((mu.T * wl.T).T, dim=0)
            X_imputed_model1 = img_zero_batch + Xm.reshape((number_of_images,3,32,32)) * (1 - img_mask_batch)

    with torch.no_grad():
        if is_conv_model2: 
            mu, lpxz, lpmz, lqzx, lpz = model2(img_zero_batch , img_mask_batch, 10)
            wl = softmax(lpxz + lpmz + lpz - lqzx)
            wl = wl.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # Compute the missing data imputation
            Xm = torch.sum(mu * wl, dim=1)
            X_imputed_model2 = img_zero_batch + Xm * (1 - img_mask_batch)
        else:
            mu, lpxz, lpmz, lqzx, lpz = model2(img_zero_batch.flatten(start_dim=1), img_mask_batch.flatten(start_dim=1), 10)
            mu = torch.exp(mu)
            # Compute the importance weights
            wl = softmax(lpxz + lpmz + lpz - lqzx)
            # Compute the missing data imputation
            Xm = torch.sum((mu.T * wl.T).T, dim=0)
            X_imputed_model2 = img_zero_batch + Xm.reshape((number_of_images,3,32,32)) * (1 - img_mask_batch)
    
     # Get mean imputation baseline
    with torch.no_grad():
        X_imputed_mean = torch.zeros_like(img_zero_batch)
        for i in range(number_of_images):
            X_imputed_mean[i], _ = mean_imputation_and_rmse(
                img_zero_batch[i], img_mask_batch[i], original_batch[i]
            )
    # Plot the images
    fig, axes = plt.subplots(number_of_images, 6, figsize=(10, 12))

    for i in range(number_of_images):
        # Original image
        axes[i, 0].imshow(original_batch[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Transformed image with blue pixels zeroed out
        axes[i, 1].imshow(img_zero_batch_copy[i].permute(1, 2, 0).numpy())  
        axes[i, 1].set_title("With missing data", wrap=True)
        axes[i, 1].axis("off")

        # Mask showing unchanged and modified pixels
        axes[i, 2].imshow(img_mask_batch[i].permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 2].set_title("Mask")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(X_imputed_model1[i].permute(1, 2, 0).numpy())
        axes[i, 3].set_title("Model trained \n on missing blue", wrap=True)
        axes[i, 3].axis("off")

        axes[i, 4].imshow(X_imputed_model2[i].permute(1, 2, 0).numpy())
        axes[i, 4].set_title("Model trained \n on missing green", wrap=True)
        axes[i, 4].axis("off")
    
        # Mean imputation baseline
        axes[i, 5].imshow(X_imputed_mean[i].permute(1, 2, 0).numpy())
        axes[i, 5].set_title("Mean imputation baseline", wrap=True)
        axes[i, 5].axis("off")
    plt.suptitle("Blue pixels missing data: Comparison between a model trained on missing blue pixels and a model trained on missing green pixels", wrap=True)

    plt.tight_layout()
    plt.savefig(f"temp/plot_comparison_plot_res_{date}.png")


if __name__=="__main__":
    # plot_images(transform_name='ZeroBlueTransform')
    # model_path = "/raid/home/detectionfeuxdeforet/caillaud_gab/mva_pgm/MVA_PGM_NotMIWAE/temp/not_miwae_2024_11_19_22_53_05_best_val_loss.pt"
    # calib_config = {'n_hidden' : 512, 'n_latent': 128, 'missing_process': 'linear', 'out_dist': 'gauss', 'transform': 'ZeroBlueTransform', 'hidden_dims' : [64,128,256]}
    # plot_images_with_imputation(model_path, is_conv_model = True, calib_config=calib_config, number_of_images=4)
    
    model_path2 = "/raid/home/detectionfeuxdeforet/caillaud_gab/mva_pgm/MVA_PGM_NotMIWAE/temp/not_miwae_2024_11_20_15_05_45_best_val_loss.pt"
    calib_config1 = {'n_hidden' : 512, 'n_latent': 128, 'missing_process': 'selfmasking', 'out_dist': 'gauss', 'transform': 'ZeroBlueTransform', 'hidden_dims' : [64,128,256]}
    model_path1 = "/raid/home/detectionfeuxdeforet/caillaud_gab/mva_pgm/MVA_PGM_NotMIWAE/temp/not_miwae_2024_11_20_18_48_28_best_val_loss.pt"
    is_conv_model1, is_conv_model2 = True, True 
    print("Using same calib_config for model1 and model2")
    plot_images_with_imputation_comparison_between_models(model_path1, is_conv_model1, calib_config1, model_path2, is_conv_model2, calib_config1, number_of_images=5)
    print("Image generated.")
