from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
from not_miwae_cifar import ZeroBlueTransform
import datetime
from torch.utils.data import Subset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

from data_imputation import compute_imputation_rmse_not_miwae, softmax
from not_miwae import get_notMIWAE, notMIWAE
from utils import seed_everything


def convert_cifar10_to_numpy(dataset, batch_size=512):
    """
    Convert a CIFAR-10 dataset to numpy arrays

    Args:
        dataset: torchvision.datasets.CIFAR10 instance
        batch_size: batch size for processing (to avoid memory issues)

    Returns:
        images: numpy array of shape (N, C, H, W)
        labels: numpy array of shape (N,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize lists to store batches
    all_images_missing = []
    all_masks = []
    all_images_true = []

    # Process dataset in batches
    for data, labels in tqdm(loader, desc="Converting to numpy"):
        images_missing, masks, images_true = data[0], data[1], data[2]  # Get the zeroed images

        images_missing = images_missing.numpy()
        masks    = masks.numpy()
        images_true = images_true.numpy()


        all_images_missing.append(images_missing)
        all_masks.append(masks)
        all_images_true.append(images_true)

    # Concatenate all batches
    all_images_missing = np.concatenate(all_images_missing, axis=0)
    all_masks= np.concatenate(all_masks, axis=0)
    all_images_true= np.concatenate(all_images_true, axis=0)

    return all_images_missing, all_masks, all_images_true



if __name__ == "__main__":

    calib_config = [{'transform': 'ZeroBlueTransform', 'dataset_size': 3,}
                    ][-1]
    if calib_config['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=True)
        ])
    else:
        raise KeyError('Transforms is not correctly defined.')

    if calib_config['dataset_size'] is not None:
        # set download to True if this is the first time you are running this file
        train_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                        download=False, transform=transform),
                           torch.arange(calib_config['dataset_size']))

        test_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                       download=False, transform=transform),
                          torch.arange(calib_config['dataset_size']))
    else:
        train_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                 download=False, transform=transform)

        test_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                download=False, transform=transform)

    # processed_images_reshaped = self.processed_images.reshape(n_samples, -1).numpy()
    all_images_missing, all_masks, all_images_true = convert_cifar10_to_numpy(train_set)
    # Initialize and fit IterativeImputer
    imputer = IterativeImputer(random_state=42, max_iter=10)
    images_imputed = imputer.fit_transform(all_images_missing)

    RMSE_iter = np.sqrt(np.sum( all_images_true- images_imputed) ** 2 * (1 - all_masks)) / np.sum(1 - all_masks)

    print("MICE, imputation RMSE on train ", RMSE_iter)

    all_images_missing_val, all_masks_val, all_images_true_val = convert_cifar10_to_numpy(test_set)
    # Initialize and fit IterativeImputer
    images_imputed_val = imputer.transform(all_images_missing_val)

    RMSE_iter_val = np.sqrt(np.sum(all_images_true_val - images_imputed_val) ** 2 * (1 - all_masks_val)) / np.sum(1 - all_masks_val)

    print("MICE, imputation RMSE on val ", RMSE_iter_val)

    estimator = RandomForestRegressor(n_estimators=100)
    imp_rf = IterativeImputer(estimator=estimator)

    images_imputed = imp_rf.fit_transform(all_images_missing)

    RMSE_iter = np.sqrt(np.sum(all_images_true - images_imputed) ** 2 * (1 - all_masks)) / np.sum(1 - all_masks)
    print("missForest, imputation RMSE on train ", RMSE_iter)
    images_imputed_val = imp_rf.transform(all_images_missing_val)
    RMSE_iter_val = np.sqrt(np.sum(all_images_true_val - images_imputed_val) ** 2 * (1 - all_masks_val)) / np.sum(1 - all_masks_val)
    print("missForest, imputation RMSE on val ", RMSE_iter_val)
