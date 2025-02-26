import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

from ..core.config import PRETRAINED_CKPT_FOLDER
from ..core.constants import NUM_JOINTS_SMPLX
from train.models.head.shape_cam_head import HMRShapeCamHead
from train.models.backbone.hrnet import hrnet_w32, hrnet_w48
from train.models.backbone.utils import get_backbone_info
from train.models.head.smplx_cam_head import SMPLXCamHead
from train.models.freia_nf import cond_realnvp_nf
from train.utils.sampling_utils import expand_annotation


class Humr(nn.Module):
    def __init__(self, backbone, img_res, pretrained_ckpt, hparams):
        super(Humr, self).__init__()
        self.hparams = hparams

        # initialize backbone, only hrnet supported right now
        assert backbone.startswith("hrnet")
        backbone, use_conv = backbone.split("-")
        pretrained_ckpt = backbone + "-" + pretrained_ckpt
        pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
        self.backbone = eval(backbone)(
            pretrained_ckpt_path=pretrained_ckpt_path,
            downsample=True,
            use_conv=(use_conv == "conv"),
        )

        if self.hparams.TRAINING.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

        context_vec_size = get_backbone_info(backbone)["n_output_channels"]

        self.num_joints = NUM_JOINTS_SMPLX
        j2d_embed_dim = 256
        self.head = HMRShapeCamHead(
            num_input_features=context_vec_size + j2d_embed_dim + 3,
        )
        self.joints_enc = nn.Sequential(
            nn.Linear(17 * 3, 512), nn.ReLU(inplace=True), nn.Linear(512, j2d_embed_dim)
        )

        self.flow = cond_realnvp_nf(
            input_dim=self.num_joints * 6,
            n_blocks=self.hparams.FREIA_NF.N_BLOCKS,
            clamp=self.hparams.FREIA_NF.CLAMP,
            fc_size=self.hparams.FREIA_NF.FC_SIZE,
            dropout=self.hparams.FREIA_NF.DROPOUT,
            cond_size=context_vec_size + 3 + j2d_embed_dim,
        )

        self.smpl = SMPLXCamHead(img_res)

    def forward(self, batch, num_samples, bbox_scale, bbox_center, img_w, img_h, fl):
        images = batch["img"]
        batch_size = images.shape[0]
        if self.hparams.TRAINING.FREEZE_BACKBONE:
            assert not self.backbone.training
        if fl is not None:
            focal_length = fl
        else:
            # Estimate focal length
            focal_length = (img_w * img_w + img_h * img_h) ** 0.5
            focal_length = focal_length.repeat(2).view(batch_size, 2)

        # Initialize cam intrinsic matrix
        cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
        cam_intrinsics[:, 0, 0] = focal_length[:, 0]
        cam_intrinsics[:, 1, 1] = focal_length[:, 1]
        cam_intrinsics[:, 0, 2] = img_w / 2.0
        cam_intrinsics[:, 1, 2] = img_h / 2.0

        # CLIFF bounding box information
        cx, cy = bbox_center[:, 0], bbox_center[:, 1]
        b = bbox_scale * 200
        bbox_info = torch.stack([cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)  # [-1, 1]
        bbox_info[:, 2] = bbox_info[:, 2] / cam_intrinsics[:, 0, 0]  # [-1, 1]
        bbox_info = bbox_info.cuda().float()

        features = self.backbone(images)
        coco_joints = batch["cond_keypoints"].view(batch_size, -1)
        coco_emb = self.joints_enc(coco_joints)

        # add 2d coco joints to head and NF
        hmr_output = self.head(features, coco_emb, bbox_info=bbox_info)
        conditioning_feats = torch.cat(
            (
                hmr_output["body_feat"],
                coco_emb,
                bbox_info,
            ),
            dim=1,
        )
        # ####### generate SMPL poses
        z = torch.randn(
            (batch_size, num_samples, self.num_joints * 6), device="cuda", dtype=torch.float32
        )
        z[:, 0] = 0  # always evaluate z0 (approximated mode)
        z = z.reshape(batch_size * num_samples, -1)
        condition = expand_annotation(conditioning_feats, num_samples)
        # inputs should be (bs*num_samples, ...)
        pred_pose_6d, log_jac_det = self.flow(z, condition, rev=True)
        pred_pose = rotation_6d_to_matrix(
            pred_pose_6d.reshape(batch_size * num_samples, self.num_joints, 6)
        ).view(batch_size, num_samples, self.num_joints, 3, 3)
        pred_pose_6d = pred_pose_6d.reshape(batch_size, num_samples, -1)
        pred_smpl_params = {
            "global_orient": pred_pose[:, :, [0]],
            "body_pose": pred_pose[:, :, 1:],
        }
        # compute log_prob:
        log_jac_det *= -1  # take inverse
        log_jac_det = log_jac_det.reshape(batch_size, num_samples)
        z = z.reshape(batch_size, num_samples, -1)
        dist_mean = torch.zeros(
            (batch_size, self.num_joints * 6), device="cuda", dtype=torch.float32
        )
        dist_sigma = torch.ones(
            (batch_size, self.num_joints * 6), device="cuda", dtype=torch.float32
        )
        gnll = torch.nn.functional.gaussian_nll_loss(
            dist_mean.unsqueeze(1), z, dist_sigma.unsqueeze(1), reduction="none"
        ).sum(dim=-1)
        loss_nll = (gnll - log_jac_det) / pred_pose_6d.shape[-1]
        log_prob = loss_nll

        pred_smpl_params["betas"] = hmr_output["pred_shape"].unsqueeze(1).repeat(1, num_samples, 1)
        pred_rot_mats = torch.cat(
            (pred_smpl_params["global_orient"], pred_smpl_params["body_pose"]), dim=2
        )
        pred_rot_mats = pred_rot_mats.reshape(batch_size * num_samples, -1, 3, 3)
        betas_pose_reproj = pred_smpl_params["betas"].reshape(batch_size * num_samples, -1)

        pred_cams = (
            hmr_output["pred_cam"]
            .unsqueeze(1)
            .repeat(1, num_samples, 1)
            .reshape(batch_size * num_samples, -1)
        )

        smpl_output = self.smpl(
            rotmat=pred_rot_mats,
            shape=betas_pose_reproj,
            cam=pred_cams,
            cam_intrinsics=expand_annotation(cam_intrinsics, num_samples),
            bbox_scale=expand_annotation(bbox_scale, num_samples),
            bbox_center=expand_annotation(bbox_center, num_samples),
            img_w=expand_annotation(img_w, num_samples),
            img_h=expand_annotation(img_h, num_samples),
        )
        smpl_output["used_focal"] = focal_length
        output = {}
        output["cam_intrinsics"] = cam_intrinsics
        output["pred_cam"] = pred_cams
        output["pred_smpl_params"] = pred_smpl_params
        output["log_prob"] = log_prob.detach()
        output["conditioning_feats"] = conditioning_feats
        output["pred_pose_6d"] = pred_pose_6d
        # fuse smpl_output with output
        output.update(smpl_output)

        # reshape all to (batchsize, num_samples, ...):
        for key, value in output.items():
            if torch.is_tensor(value) and value.shape[0] == batch_size * num_samples:
                old_shape = value.shape
                output[key] = value.reshape(batch_size, num_samples, *old_shape[1:])
        return output
