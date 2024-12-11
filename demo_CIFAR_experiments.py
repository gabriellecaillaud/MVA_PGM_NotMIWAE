
""" This file is intended to describe each step of our work on the CIFAR-10 dataset.
Since training a VAE on the CIFAR-10 dataset takes time and computationnal ressources,
we did our work on the CentraleSupélec's DGX, which is made of 4 A100 available to students.
We are not able to run notebooks on the DGX, so we describe our experiments in this script instead of a notebook."""

# Imports
import datetime
import matplotlib.pyplot as plt
from code.code_CIFAR_experiments.baselines import mean_imputation_and_rmse
from code.common.data_imputation import softmax
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms

from code.code_CIFAR_experiments.model import ConvNotMIWAE
from code.code_CIFAR_experiments.train import train_conv_notMIWAE_on_cifar10
from code.code_CIFAR_experiments.transforms import ZeroBlueTransform
from code.common.utils import seed_everything


######### HYPER-PARAMETERS SETTING #########

# All of the parameters for the model in the calib_config dictionnary
# Putting every hyperparameter in one single dictionnary enables us to keep track more easily of the experiments we conducted

calib_config = [
    {'model': 'not_miwae', 'lr': 1e-3, 'epochs': 100, 'pct_start': 0.1, 'final_div_factor': 1e4, 'batch_size': 8,
     'n_hidden': 512, 'n_latent': 128, 'missing_process': 'selfmasking', 'weight_decay': 0, 'betas': (0.9, 0.999),
     'random_seed': 0, 'out_dist': 'gauss',
     'hidden_dims': [64, 128, 256]},
][-1]
print(f"calib_config:{calib_config}")

######### DATA PREPARATION #########


seed_everything(calib_config['random_seed'])  # Reproducibility

# In this script, we will remove blue pixels for which the intensity is superior to red and green.
transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
        ])

# If you are running this file for the first time, please use download = True to download the CIFAR-10 dataset.
train_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                         download=False, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                        download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=calib_config['batch_size'],
                                           shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=calib_config['batch_size'],
                                          shuffle=False, num_workers=2)

######### TRAINING #########

date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(f"Current timestamp : {date}")

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Defining the model
model = ConvNotMIWAE(n_latent=calib_config['n_latent'],
                     activation=nn.ReLU(),
                     out_activation=nn.Sigmoid(),
                     hidden_dims=calib_config['hidden_dims'],
                     missing_process=calib_config['missing_process'])

model.to(device)
print(f"Number of parameters in the model: {sum(p.numel() if p.requires_grad else 0 for p in model.parameters())}")
# Defining optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=calib_config['lr'], weight_decay=calib_config['weight_decay'],
                             betas=calib_config['betas'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=calib_config['lr'],
                                                epochs=calib_config['epochs'],
                                                steps_per_epoch=len(train_loader),
                                                pct_start=calib_config['pct_start'],
                                                )
# Training the model
train_conv_notMIWAE_on_cifar10(model=model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer,
                               scheduler=scheduler, num_epochs=calib_config['epochs'], total_samples_x_train=10,
                               device=device, date=date)
# The model is saved in a /temp/ directory.

######### PLOTTING RESULTS #########

# Usually, to plot the results, we load the saved model and use the functions defined in the
# file code/code_CIFAR_experiments/plot.py
# Here, for demonstrations purposes, we put the code in this script.
# In this script we plot the results of the model we trained on.
# Other functions in the code/code_CIFAR_experiments/plot.py allow comparison between models.
# You can check the generated images in our pdf report.

number_of_images_to_plot = 5
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=number_of_images_to_plot,
                                              shuffle=True, num_workers=2)

model.eval()
data = next(iter(test_dataloader))
img_zero_batch, img_mask_batch, original_batch = data[0]
img_zero_batch_copy = img_zero_batch.detach().clone()
with torch.no_grad():
    mu, lpxz, lpmz, lqzx, lpz = model(img_zero_batch, img_mask_batch, 10)
    wl = softmax(lpxz + lpmz + lpz - lqzx)
    wl = wl.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # Compute the missing data imputation
    Xm = torch.sum(mu * wl, dim=1)
    X_imputed = img_zero_batch + Xm * (1 - img_mask_batch)

# Results with mean imputation baseline
with torch.no_grad():
    X_imputed_mean = torch.zeros_like(img_zero_batch)
    for i in range(number_of_images_to_plot):
        X_imputed_mean[i], _ = mean_imputation_and_rmse(
            img_zero_batch[i], img_mask_batch[i], original_batch[i]
        )

# Plot the images
fig, axes = plt.subplots(number_of_images_to_plot, 5, figsize=(10, 12))

for i in range(number_of_images_to_plot):
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
    axes[i, 2].set_title("Modification Mask")
    axes[i, 2].axis("off")

    # Results after imputation
    axes[i, 3].imshow(X_imputed[i].permute(1, 2, 0).numpy())
    axes[i, 3].set_title("After imputation")
    axes[i, 3].axis("off")

    # Results with mean baseline
    axes[i, 3].imshow(X_imputed_mean[i].permute(1, 2, 0).numpy())
    axes[i, 3].set_title("Using Mean baseline")
    axes[i, 3].axis("off")
