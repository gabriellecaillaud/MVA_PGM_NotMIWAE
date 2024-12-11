import torch

""" 
This file defines the different ways we introduced missing data into the MNAR dataset. 
"""


class ZeroBlueTransform:
    def __init__(self, do_flatten=True):
        print("Missing data type: blue channel put to zero when blue is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where blue is the dominant color channel
        blue_dominant_mask = (img[2, :, :] > img[0, :, :]) & (img[2, :, :] > img[1, :, :])

        # Zero out pixels where blue is not dominant
        img_zero = img.clone()
        img_zero[2, blue_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[2, blue_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img


class ZeroGreenTransform:
    def __init__(self, do_flatten=True):
        print("Missing data type: GREEN channel put to zero when GREEN is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where GREEN is the dominant color channel
        green_dominant_mask = (img[1, :, :] > img[0, :, :]) & (img[1, :, :] > img[2, :, :])

        # Zero out pixels where green is not dominant
        img_zero = img.clone()
        img_zero[1, green_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[1, green_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img


class ZeroRedTransform:
    def __init__(self, do_flatten=True):
        print("Missing data type: RED channel put to zero when RED is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where GREEN is the dominant color channel
        red_dominant_mask = (img[0, :, :] > img[1, :, :]) & (img[0, :, :] > img[2, :, :])

        # Zero out pixels where green is not dominant
        img_zero = img.clone()
        img_zero[0, red_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[0, red_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img


class ZeroPixelWhereBlueTransform:
    def __init__(self, do_flatten=True):
        print("Type of missing data: all channels put to zero when blue is the most intense color")
        self.do_flatten = do_flatten

    def __call__(self, img):
        # Normalize to [0, 1] (assuming img is in range [0, 255])

        # Create a mask for pixels where blue is the dominant color channel
        blue_dominant_mask = (img[2, :, :] > img[0, :, :]) & (img[2, :, :] > img[1, :, :])

        # Zero out pixels where blue is not dominant
        img_zero = img.clone()
        img_zero[:, blue_dominant_mask] = 0

        # Create img_mask (1 if unchanged, 0 if modified)
        img_mask = torch.ones_like(img)
        img_mask[:, blue_dominant_mask] = 0

        if self.do_flatten:
            # Flatten tensors if needed (CIFAR-10 may not require flattening)
            return img_zero.flatten(), img_mask.flatten(), img.flatten()
        else:
            return img_zero, img_mask, img
