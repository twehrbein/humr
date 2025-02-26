"""
This file contains functions that are used to perform data augmentation.
"""
import cv2
import torch
from skimage.transform import rotate
import numpy as np

from ..core import constants


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    assert rot == 0  # not supported for now
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]

    if not rot == 0:
        # Remove padding
        new_img = rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    # resize image:
    if res[0] > new_img.shape[0] and res[1] > new_img.shape[1]:
        new_img = cv2.resize(new_img, (res[1], res[0]), interpolation=cv2.INTER_LINEAR)
    else:
        new_img = cv2.resize(new_img, (res[1], res[0]), interpolation=cv2.INTER_AREA)
    return new_img


def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(center, scale, crop_scale_factor, axis="all"):
    """
    center: bbox center [x,y]
    scale: bbox height / 200
    crop_scale_factor: amount of cropping to be applied
    axis: axis which cropping will be applied
        "x": center the y axis and get random crops in x
        "y": center the x axis and get random crops in y
        "all": randomly crop from all locations
    """
    orig_size = int(scale * 200.0)
    ul = (center - (orig_size / 2.0)).astype(int)

    crop_size = int(orig_size * crop_scale_factor)

    if axis == "all":
        h_start = np.random.rand()
        w_start = np.random.rand()
    elif axis == "x":
        h_start = np.random.rand()
        w_start = 0.5
    elif axis == "y":
        h_start = 0.5
        w_start = np.random.rand()
    else:
        raise ValueError(f"axis {axis} is undefined!")

    x1, y1, x2, y2 = get_random_crop_coords(
        height=orig_size,
        width=orig_size,
        crop_height=crop_size,
        crop_width=crop_size,
        h_start=h_start,
        w_start=w_start,
    )
    scale = (y2 - y1) / 200.0
    center = ul + np.array([(y1 + y2) / 2, (x1 + x2) / 2])
    return center, scale


def img_torch_to_np_denormalize(
    images,
):
    IMG_NORM_MEAN = torch.tensor(constants.IMG_NORM_MEAN, device=images.device).reshape(1, 3, 1, 1)
    IMG_NORM_STD = torch.tensor(constants.IMG_NORM_STD, device=images.device).reshape(1, 3, 1, 1)
    images = (images * IMG_NORM_STD) + IMG_NORM_MEAN
    imges_np = images.permute((0, 2, 3, 1)).cpu().numpy()
    imges_np = np.clip(np.rint(imges_np * 255), 0, 255).astype(np.uint8)
    return imges_np


def read_img(img_fn):
    return cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)


def boxes_2_cs(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2-x1, y2-y1
    cx, cy = x1+w/2, y1+h/2
    size = np.stack([w, h]).max(axis=0)

    centers = np.stack([cx, cy], axis=1)
    scales = size / 200
    return centers, scales
