import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np

from ..core.constants import (
    IMG_NORM_MEAN,
    IMG_NORM_STD,
)
from train.utils.image_utils import boxes_2_cs, crop


class DetectDataset(Dataset):
    """
    Detection Dataset Class - Handles data loading from detections.
    """

    def __init__(self, img, boxes, crop_size=224, dilate=1.2, img_focal=None, img_center=None):
        super(DetectDataset, self).__init__()

        self.img = img
        self.crop_size = crop_size
        self.orig_shape = img.shape[:2]
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)

        self.boxes = boxes
        self.box_dilate = dilate
        # boxes to center & scale format
        self.centers, self.scales = boxes_2_cs(boxes)

        if img_focal is None:
            self.img_focal = self.est_focal(self.orig_shape)
        else:
            self.img_focal = img_focal

        if img_center is None:
            self.img_center = self.est_center(self.orig_shape)
        else:
            self.img_center = img_center

    def __getitem__(self, index):
        item = {}
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]
        img_focal = self.img_focal
        img_center = self.img_center

        img = crop(self.img, center, scale, [self.crop_size, self.crop_size])
        img = np.transpose(img.astype("float32"), (2, 0, 1)) / 255.0
        img = torch.from_numpy(img).float()
        item["img"] = self.normalize_img(img)
        # generate image crop for 2D pose detector (256x192)
        img2 = crop(self.img, center, scale, [256, 256])
        img2 = np.transpose(img2.astype("float32"), (2, 0, 1)) / 255.0
        img2 = img2[:, :, 32:-32]
        img2 = torch.from_numpy(img2).float()
        item["img2"] = self.normalize_img(img2)

        item["scale"] = torch.tensor(scale).float()
        item["center"] = torch.tensor(center).float()
        item["img_focal"] = torch.tensor(img_focal).float()
        item["img_center"] = torch.tensor(img_center).float()
        item["orig_shape"] = torch.tensor(self.orig_shape).float()

        return item

    def __len__(self):
        return len(self.boxes)

    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal

    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w / 2.0, h / 2.0])
        return center
