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


def convert_cifar10_to_numpy(dataset, batch_size=8):
    """
    Convert a CIFAR-10 dataset to numpy arrays

    Args:
        dataset: torchvision.datasets.CIFAR10 instance
        batch_size: batch size for processing (to avoid memory issues)

    Returns:
        images: numpy array of shape (N, C, H, W)
        labels: numpy array of shape (N,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # Initialize lists to store batches
    n_samples = len(dataset)
    n_features = 3072  # CIFAR-10 flattened size (32*32*3)
    
    # Pre-allocate numpy arrays
    images_missing = np.zeros((n_samples, n_features), dtype=np.float32)
    masks = np.zeros((n_samples, n_features), dtype=np.float32)
    images_true = np.zeros((n_samples, n_features), dtype=np.float32)

    # Process dataset in batches
    start_idx = 0
    for data, _ in loader:
        print(f"{start_idx=}")
        # Get current batch size (last batch might be smaller)
        current_batch_size = data[0].shape[0]
        end_idx = start_idx + current_batch_size
        
        # Unpack the tuple from ZeroBlueTransform and convert to numpy
        images_missing[start_idx:end_idx] = data[0].numpy()
        masks[start_idx:end_idx] = data[1].numpy()
        images_true[start_idx:end_idx] = data[2].numpy()
        
        # Update the start index for the next batch
        start_idx = end_idx
    
    print("Inside function torch to numpy")
    return images_missing, masks, images_true

def mean_imputation_and_rmse(img_zero, img_mask, img_original):
    """
    Replace missing blue channel values with the mean of remaining blue values
    and compute RMSE between imputed and original values.
    
    Args:
        img_zero (torch.Tensor): Image tensor with zeroed-out blue values
        img_mask (torch.Tensor): Mask indicating which values were zeroed (0) vs kept (1)
        img_original (torch.Tensor): Original image tensor before modification
        
    Returns:
        tuple: (imputed_img, rmse)
            - imputed_img: Image tensor with imputed values
            - rmse: Root Mean Square Error between original and imputed values
    """
    imputed_img = img_zero.detach().clone()
    
    # If the tensors are flattened, reshape them to [channels, height, width]
    if len(img_zero.shape) == 1:
        size = int(np.cbrt(len(img_zero) / 3))  # Calculate original image size
        img_zero = img_zero.reshape(3, size, size)
        img_mask = img_mask.reshape(3, size, size)
        img_original = img_original.reshape(3, size, size)
        imputed_img = imputed_img.reshape(3, size, size)
    
    # Get blue channel components
    blue_channel = img_zero[2]
    blue_mask = img_mask[2]
    
    # Calculate mean of non-zero blue values
    valid_blue_values = blue_channel[blue_mask == 1]
    if len(valid_blue_values) > 0:
        blue_mean = valid_blue_values.mean()
    else:
        blue_mean = 0.0  # Fallback if all blue values were zeroed
    
    # Replace zeroed values with the mean
    blue_channel[blue_mask == 0] = blue_mean
    
    # Put the imputed blue channel back
    imputed_img[2] = blue_channel
    
    # Calculate RMSE only for the modified blue values
    modified_positions = (img_mask[2] == 0)
    if modified_positions.sum() > 0:
        squared_errors = (imputed_img[2][modified_positions] - 
                         img_original[2][modified_positions]) ** 2
        rmse = torch.sqrt(squared_errors.mean())
    else:
        rmse = torch.tensor(0.0)
    
    # Reshape back to original format if needed
    if len(img_zero.shape) == 1:
        imputed_img = imputed_img.flatten()
    
    return imputed_img, rmse

# Example usage with a dataloader
def evaluate_mean_imputation(dataloader):
    """
    Evaluate the mean imputation method over a dataset.
    
    Args:
        dataloader: PyTorch DataLoader containing the dataset
        
    Returns:
        float: Average RMSE across the dataset
    """
    total_rmse = 0.0
    num_samples = 0
    
    for batch in dataloader:
        # Assuming the dataloader returns tuples of (img_zero, img_mask, img_original)
        img_zero, img_mask, img_original = batch[0]
        _, rmse = mean_imputation_and_rmse(
                img_zero, img_mask, img_original
                )
        total_rmse += rmse.item()
        num_samples += 1
    
    avg_rmse = total_rmse / num_samples
    return avg_rmse

if __name__ == "__main__":

    calib_config = [{'transform': 'ZeroBlueTransform', 'dataset_size': None}
                    ][-1]
    if calib_config['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
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


    avg_rmse  =evaluate_mean_imputation(test_set)
    print(f"Baseline on test set: Average RMSE when replacing missing values with the mean : {avg_rmse}")
    # processed_images_reshaped = self.processed_images.reshape(n_samples, -1).numpy() 
    # all_images_missing, all_masks, all_images_true = convert_cifar10_to_numpy(train_set)
    # # Initialize and fit IterativeImputer
    # print("Before definition of IterativeImputer")
    # imputer = IterativeImputer(random_state=42, max_iter=10)
    # print("Before fitting the imputer")
    # images_imputed = imputer.fit_transform(all_images_missing)

    # RMSE_iter = np.sqrt(np.sum( all_images_true- images_imputed) ** 2 * (1 - all_masks)) / np.sum(1 - all_masks)

    # print("MICE, imputation RMSE on train ", RMSE_iter)

    # all_images_missing_val, all_masks_val, all_images_true_val = convert_cifar10_to_numpy(test_set)
    # # Initialize and fit IterativeImputer
    # images_imputed_val = imputer.transform(all_images_missing_val)

    # RMSE_iter_val = np.sqrt(np.sum(all_images_true_val - images_imputed_val) ** 2 * (1 - all_masks_val)) / np.sum(1 - all_masks_val)
    # print("MICE, imputation RMSE on val implementation 1", RMSE_iter_val)

    # RMSE_iter_val = np.sqrt(np.sum((all_images_true_val - images_imputed_val)**2 * (1 - all_masks_val))) / np.sum(1 - all_masks_val)
    # print("MICE, imputation RMSE on val implementation Claude", RMSE_iter_val)

    # estimator = RandomForestRegressor(n_estimators=100)
    # imp_rf = IterativeImputer(estimator=estimator)

    # images_imputed = imp_rf.fit_transform(all_images_missing)

    # RMSE_iter = np.sqrt(np.sum(all_images_true - images_imputed) ** 2 * (1 - all_masks)) / np.sum(1 - all_masks)
    # print("missForest, imputation RMSE on train ", RMSE_iter)
    # images_imputed_val = imp_rf.transform(all_images_missing_val)
    # RMSE_iter_val = np.sqrt(np.sum(all_images_true_val - images_imputed_val) ** 2 * (1 - all_masks_val)) / np.sum(1 - all_masks_val)
    # print("missForest, imputation RMSE on val ", RMSE_iter_val)

