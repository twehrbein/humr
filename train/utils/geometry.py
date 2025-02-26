import torch


def convert_pare_to_full_img_cam(pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    tz = 2 * focal_length / (bbox_height * s)
    cx = 2 * (bbox_center[:, 0] - (img_w / 2.0)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.0)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t


def perspective_projection(points, cam_intrinsics, rotation=None, translation=None):
    K = cam_intrinsics
    if rotation is not None:
        points = torch.einsum("bij,bkj->bki", rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points.float())
    return projected_points[:, :, :-1]
