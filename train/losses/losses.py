import torch
import torch.nn as nn

from train.core.constants import (
    SMPLX_TO_COCO25,
    COCO25_TO_COCO17,
)


# https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def batch_mmd(x, y):
    # x (bs, #samples, dim), y (bs, #samples, dim)
    xx = torch.matmul(x, x.transpose(1, 2))
    yy = torch.matmul(y, y.transpose(1, 2))
    zz = torch.matmul(x, y.transpose(1, 2))
    # (bs, #samples, #samples)

    rx = torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx)
    ry = torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(xx)

    dxx = rx.transpose(1, 2) + rx - 2.0 * xx
    dyy = ry.transpose(1, 2) + ry - 2.0 * yy
    dxy = rx.transpose(1, 2) + ry - 2.0 * zz

    XX, YY, XY = (torch.zeros_like(xx), torch.zeros_like(xx), torch.zeros_like(xx))

    # kernel computation over bandwidth range
    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx) ** -1
        YY += a**2 * (a**2 + dyy) ** -1
        XY += a**2 * (a**2 + dxy) ** -1

    return torch.mean(XX + YY - 2.0 * XY, dim=(1, 2))


class Keypoint2DLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError("Unsupported loss function")

    def forward(
        self,
        pred_keypoints_2d: torch.Tensor,
        gt_keypoints_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2]
            containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3]
            containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :, :-1])).sum(
            dim=(2, 3)
        )
        return loss


class HumrLoss(nn.Module):
    def __init__(self, hparams):
        super(HumrLoss, self).__init__()
        self.criterion_mse_noreduce = nn.MSELoss(reduction="none")
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type="l1")
        self.hparams = hparams

        self.num_joints = 24
        self.hm_min_th = self.hparams.TRAINING.HM_SAMPLE_MIN_TH
        self.hm_max_th = self.hparams.TRAINING.HM_SAMPLE_MAX_TH

    def forward(self, pred, gt, curr_epoch=0):
        center = gt["center"][:, None]
        scale = gt["scale"][:, None, None] * 200
        batch_size = center.shape[0]
        num_hm_samples = self.hparams.TRAINING.NUM_TRAIN_SAMPLES - 1
        hm_samples = gt["hm_samples_orig"][:, :num_hm_samples]  # (B, 25, 17, 2)
        assert hm_samples.shape[1] == num_hm_samples
        # normalize to [-1, 1]
        hm_samples_normed = 2 * (hm_samples - center[:, None]) / scale[:, None]
        max_confs = gt["hm_max_confs"]  # (bs, 17)
        gt_betas = gt["betas"]

        gt_keypoints_2d = gt["keypoints_orig"]  # (B, 24, 3)
        assert gt_keypoints_2d[:, :, 2].sum() == batch_size * self.num_joints
        gt_keypoints_2d_normed = gt_keypoints_2d.clone()
        gt_keypoints_2d_normed[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] - center) / scale
        gt_coco_normed = gt["gtkps_coco"][:, :, :2]  # already normalized to [-1, 1] (B, 17, 2)

        pred_smpl_params = pred["pred_smpl_params"]
        num_samples = pred_smpl_params["body_pose"].shape[1]
        pred_pose_6d = pred["pred_pose_6d"]
        pred_keypoints_2d = pred["joints2d"]  # (B, S, 127, 2)
        pred_keypoints_2d_normed = pred_keypoints_2d.clone()
        pred_keypoints_2d_normed[:, :, :, :2] = (
            2 * (pred_keypoints_2d[:, :, :, :2] - center[:, None]) / scale[:, None]
        )
        pred_coco17_normed = pred_keypoints_2d_normed[:, :, SMPLX_TO_COCO25][:, :, COCO25_TO_COCO17]

        # ########### compute 2d reproj loss for approximated mode prediction
        n_j_samples = pred_keypoints_2d_normed.shape[1]
        pred_keypoints_2d_normed = pred_keypoints_2d_normed[:, :, : self.num_joints]  # (B, S, N, ...)
        loss_keypoints_2d = self.keypoint_2d_loss(
            pred_keypoints_2d_normed, gt_keypoints_2d_normed.unsqueeze(1).expand(-1, n_j_samples, -1, -1)
        )
        # The first item of the second dimension always corresponds to the mode pred
        loss_keypoints_2d_mode = (
            loss_keypoints_2d[:, 0].sum() / batch_size
        ) * self.hparams.LOSS_WEIGHTS.KEYPOINTS_2D_MODE

        # ################# MMD computation
        # only use heatmap samples in specific cases, otherwise fill joints with gt.
        # 2d detector confidence value in specific range:
        valid_samples_mask = torch.logical_and(max_confs < self.hm_max_th, max_confs > self.hm_min_th)
        # only highly articulated joints
        valid_samples_mask = torch.logical_and(gt["valid_sample_joints"], valid_samples_mask)
        # gt joint should be inside the crop, otherwise detections are probably faulty
        valid_samples_mask = torch.logical_and(gt["coco_gt_bbox_inside"], valid_samples_mask)
        valid_samples_mask = torch.logical_and(gt["coco_inside_square_crop"], valid_samples_mask)[
            :, None
        ]
        valid_samples_mask = valid_samples_mask.expand(-1, num_hm_samples, -1)  # (B, S, 17)
        # use gt 2d joints otherwise
        target_samples = gt_coco_normed[:, None].repeat(1, num_hm_samples, 1, 1)
        # fill in valid heatmap samples
        target_samples[valid_samples_mask] = hm_samples_normed[valid_samples_mask]

        # do not penalize joints where gt is outside of crop
        invalid_target = torch.logical_and(~gt["valid_sample_joints"], max_confs < 0.5)
        valid_target = ~torch.logical_or(~gt["coco_inside_square_crop"], invalid_target)[
            :, None
        ].squeeze()

        # always use gt coco joints as one of the MMD target samples
        target_samples[:, [0]] = gt_coco_normed.unsqueeze(1)
        target_samples = target_samples.transpose(1, 2).reshape(batch_size * 17, num_hm_samples, 2)
        # do not penalize mode prediction here, only the random NF samples
        pred_samples = (
            pred_coco17_normed[:, 1:].transpose(1, 2).reshape(batch_size * 17, num_hm_samples, 2)
        )
        mmd = batch_mmd(pred_samples, target_samples).reshape(batch_size, 17)
        loss_samples_2d_exp = (
            mmd * valid_target
        ).mean() * self.hparams.LOSS_WEIGHTS.KEYPOINTS_2D_HM_SAMPLE

        # ######### MASK LOSS COMPUTATION
        # need pred 2d joints in [0, IMG_RES] space
        pred_2d_mask_space = (pred_coco17_normed + 1.0) * self.hparams.DATASET.IMG_RES * 0.5
        pred_2d_mask_space = pred_2d_mask_space.reshape(batch_size, num_samples * 17, 2)
        mask_indices = gt["mask_indices"]
        # compute distances to all mask pixels
        distances = torch.cdist(pred_2d_mask_space, mask_indices, p=2)
        shortest_mask_dist, _ = torch.min(distances, dim=-1)  # (bs, num_hypo * num_joints)
        # min distance is ~0.71, 0.71 ~ sqrt(0.5² + 0.5²)
        shortest_mask_dist[shortest_mask_dist < 0.71] = 0
        # only compute loss for joints that should be invisible but are visible
        outside_mask = shortest_mask_dist.reshape(batch_size, num_samples, 17) > 0
        joint_invis = torch.logical_or(max_confs < 0.5, ~gt["coco_inside_square_crop"])
        # check if predicted joints are inside img crop
        x_preds = pred_2d_mask_space[:, :, 0]
        y_preds = pred_2d_mask_space[:, :, 1]
        inside_img_w = torch.logical_and(
            x_preds >= 0, x_preds <= self.hparams.DATASET.IMG_RES - 1
        )
        inside_img_h = torch.logical_and(
            y_preds >= 0, y_preds <= self.hparams.DATASET.IMG_RES - 1
        )
        inside_img = torch.logical_and(inside_img_w, inside_img_h).reshape(
            batch_size, num_samples, 17
        )
        # boolean mask which encodes for which joints hypotheses mask loss is computed
        l_masking = (
            torch.logical_and(outside_mask, inside_img)
            * joint_invis[:, None]
            * gt["valid_mask"][:, None, None]
        )  # (bs, 26, 17)
        # instead of minimizing the minimum distance to all mask pixels,
        # minimize the minimum distance to the union of valid heatmap samples and gt 2d joints
        # hm samples that lie outside of person mask aren't valid and not used
        samples_outside_mask = gt["samples_outside_mask"][:, :num_hm_samples]
        # hack such that these samples are not selected
        hm_samples_normed[samples_outside_mask] = 1000000
        possible_targets = torch.cat((hm_samples_normed, gt_coco_normed.unsqueeze(1)), dim=1)
        dists = torch.cdist(
            pred_coco17_normed.reshape(batch_size, -1, 2),
            possible_targets.reshape(batch_size, -1, 2),
            p=1,
        )  # (bs, nf_samples*17, target_samples * 17)
        min_l1_distances, _ = torch.min(dists, dim=-1)
        min_l1_distances = min_l1_distances.reshape(batch_size, num_samples, 17)
        loss_mask = (min_l1_distances * l_masking).sum(dim=-1)

        loss_mask = (
            loss_mask[:, 1:].sum() / (batch_size * (num_samples - 1))
        ) * self.hparams.LOSS_WEIGHTS.LOCATION_MASK

        # loss on SMPL shape parameters
        gt_betas = gt_betas.unsqueeze(1).expand(-1, num_samples, -1)
        loss_betas = self.criterion_mse_noreduce(pred_smpl_params["betas"], gt_betas)
        loss_betas_mode = (
            loss_betas[:, 0].sum() / batch_size
        ) * self.hparams.LOSS_WEIGHTS.BETAS_MODE

        # loss on pred cam scale
        pred_cam = pred["pred_cam"]  # (B, S, 3)
        loss_cam = ((torch.exp(-pred_cam[:, :, 0] * 10)) ** 2).mean()

        # Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3)
        loss_pose_6d = (
            torch.matmul(pred_pose_6d, pred_pose_6d.permute(0, 2, 1))
            - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)
        ) ** 2
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1)
        loss_pose_6d_mode = loss_pose_6d[:, 0].mean() * self.hparams.LOSS_WEIGHTS.ORTHOGONAL
        loss_pose_6d_exp = loss_pose_6d[:, 1:].mean() * self.hparams.LOSS_WEIGHTS.ORTHOGONAL

        loss_dict = {
            "loss/loss_keypoints_mode": loss_keypoints_2d_mode,
            "loss/loss_samples2d_exp": loss_samples_2d_exp,
            "loss/loss_betas_mode": loss_betas_mode,
            "loss/loss_pose_6d_mode": loss_pose_6d_mode,
            "loss/loss_pose_6d_exp": loss_pose_6d_exp,
            "loss/loss_cam": loss_cam,
            "loss/loss_mask": loss_mask,
        }

        loss = sum(loss for loss in loss_dict.values())
        return loss, loss_dict
