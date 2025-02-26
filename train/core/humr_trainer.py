import os
import cv2
import torch
import pickle
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import random
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from . import config
from ..dataset.dataset import DatasetHMR
from train.models.head.smplx_local import SMPLX
from train.models.head.smpl_head import SMPL
from .constants import (
    H36M_TO_J14,
    SMPL45_TO_COCO25,
    COCO25_TO_COCO17,
    NUM_JOINTS_SMPLX,
)
from ..losses.losses import HumrLoss
from train.utils.eval_utils import (
    compute_similarity_transform_torch,
)
from train.utils.vis_utils import (
    viz_mesh_xyz_std_dev,
    scatter_joint_hypos,
    plot_multiple_hypos,
    viz_3d_samples,
)
from train.utils.image_utils import img_torch_to_np_denormalize, read_img
from train.utils.sampling_utils import vertex_diversity_from_samples, expand_annotation
from train.utils.geometry import perspective_projection
from train.models.humr_model import Humr


class HumrTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super(HumrTrainer, self).__init__()
        self.hparams.update(hparams)

        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)
        self.smpl_neutral = SMPL(config.SMPL_MODEL_DIR, gender="neutral", create_transl=False)
        self.smpl_faces = self.smpl_neutral.faces
        smplx2smpl = pickle.load(open(config.SMPLX2SMPL, "rb"))
        self.smplx2smpl = torch.tensor(smplx2smpl["matrix"], dtype=torch.float32).to("cuda")
        self.register_buffer(  # regresses 17 H36M joints
            "J_regressor", torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()
        self.val_ds = self.val_dataset("test" if hparams.RUN_TEST else "val")

        self.model = Humr(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )

        self.loss_fn = HumrLoss(hparams=self.hparams)

        self.num_train_samples = self.hparams.TRAINING.NUM_TRAIN_SAMPLES
        assert self.num_train_samples >= 2  # z0 + minimum 1 random sample
        self.num_val_samples = self.hparams.TRAINING.NUM_TEST_SAMPLES
        self.smpl_noise_ratio = self.hparams.TRAINING.SMPL_PARAM_NOISE_RATIO
        self.validation_step_outputs = []
        self.val_viz_batch_idx = []
        self.NUM_VIZ_BATCHES = hparams.TRAINING.NUM_VIZ

    def training_step(self, batch, batch_nb, dataloader_idx=0):
        gt_betas = batch["betas"]
        bbox_scale = batch["scale"]
        bbox_center = batch["center"]
        img_h = batch["orig_shape"][:, 0]
        img_w = batch["orig_shape"][:, 1]
        fl = batch["focal_length"]
        gt_pose = batch["pose"]
        batch_size = gt_pose.shape[0]

        # could be moved to dataset.py
        with torch.no_grad():
            gt_out = self.smplx(
                betas=gt_betas,
                body_pose=gt_pose[:, 3 : NUM_JOINTS_SMPLX * 3],
                global_orient=gt_pose[:, :3],
                pose2rot=True,
            )
        batch["vertices"] = gt_out.vertices
        batch["joints3d"] = gt_out.joints

        pred = self(
            batch,
            self.num_train_samples,
            bbox_center=bbox_center,
            bbox_scale=bbox_scale,
            img_w=img_w,
            img_h=img_h,
            fl=fl,
        )

        # compute NLL loss of gt pose:
        # transform gt smpl pose to correct 6D representation
        gt_6D_pose = matrix_to_rotation_6d(axis_angle_to_matrix(gt_pose.reshape(-1, 3))).reshape(
            batch_size, 1, -1
        )
        smpl_params = dict()
        # add some noise to annotations at training time to prevent overfitting
        smpl_params["body_pose"] = gt_6D_pose[:, :, 6:] + self.smpl_noise_ratio * torch.randn_like(
            gt_6D_pose[:, :, 6:]
        )
        smpl_params["global_orient"] = gt_6D_pose[
            :, :, :6
        ] + self.smpl_noise_ratio * torch.randn_like(gt_6D_pose[:, :, :6])
        gt_pose_6d = torch.cat((smpl_params["global_orient"], smpl_params["body_pose"]), dim=-1)

        # compute negative log-likelihood loss of gt sample
        gt_sample, log_jac_det = self.model.flow(gt_pose_6d.squeeze(), pred["conditioning_feats"])
        target_mean = torch.zeros_like(gt_sample)
        target_sigma = torch.ones_like(gt_sample)
        gnll = torch.nn.functional.gaussian_nll_loss(
            target_mean, gt_sample, target_sigma, reduction="none"
        ).sum(dim=1)
        loss_nll = torch.mean(gnll - log_jac_det) / gt_sample.shape[1]
        loss_nll = loss_nll * self.hparams.LOSS_WEIGHTS.NLL

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch, curr_epoch=self.current_epoch)
        loss = loss + loss_nll
        loss_dict["loss/loss_nll"] = loss_nll

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log_dict(loss_dict)
        return {"loss": loss}

    def forward(self, batch, num_samples, bbox_center, bbox_scale, img_w, img_h, fl=None):
        return self.model(
            batch,
            num_samples,
            bbox_center=bbox_center,
            bbox_scale=bbox_scale,
            img_w=img_w,
            img_h=img_h,
            fl=fl,
        )

    def on_train_epoch_start(self):
        if self.hparams.TRAINING.FREEZE_BACKBONE:
            self.model.backbone.eval()

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss_epoch"].item()
        logger.info(
            f"Training loss epoch: {train_loss:.3f}"
        )

    def on_validation_epoch_start(self, dataloaders=None):
        # randomly select samples to visualize
        self.val_viz_batch_idx.clear()
        if dataloaders is None:
            dataloaders = self.trainer.val_dataloaders
        for dataloader in dataloaders:
            max_val_batches = len(dataloader)
            self.val_viz_batch_idx.append(
                random.sample(range(max_val_batches), self.NUM_VIZ_BATCHES)
            )

    def on_test_epoch_start(self):
        self.on_validation_epoch_start(dataloaders=self.trainer.test_dataloaders)

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        logger.info(f"***** Epoch {self.current_epoch} *****")
        val_log = {}
        # get all metrics
        all_metric_keys = set()
        for metrics in outputs:
            all_metric_keys = all_metric_keys | metrics.keys()
        all_metric_keys = list(all_metric_keys)
        all_metric_keys.sort()

        # compute mean of and log every metric
        for metric in all_metric_keys:
            # if values of metric are tuples, need to compute weighted average
            metric_value = 0.0
            total_weight = 0.0
            is_tuple = False
            for x in outputs:
                if metric in x and isinstance(x[metric], tuple):
                    is_tuple = True
                    value = x[metric][0] * x[metric][1]
                    if not np.isnan(value):
                        metric_value += value
                        total_weight += x[metric][1]
            if total_weight != 0.0:
                metric_value /= total_weight
            else:
                metric_value = 0.0
            if not is_tuple:
                metric_value = np.hstack([x[metric] for x in outputs if metric in x]).mean()
            # transform "ds-name_metric-name" to "ds-name/metric-name" for logging
            log_metric = metric.replace("_", "/", 1)
            val_log[log_metric] = metric_value
            logger.info(f"{metric}: {metric_value:.2f}")

        self.log_dict(val_log)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_nb, dataloader_idx=0):
        return self.validation_step(batch, batch_nb, dataloader_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD,
        )
        return optimizer

    def train_dataset(self):
        options = self.hparams.DATASET
        dataset_names = options.TRAIN_DS.split("_")
        dataset_list = [DatasetHMR(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)
        logger.info(f"Total train dataset num samples {len(train_ds)}")
        return train_ds

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
            drop_last=True,
            persistent_workers=False,
        )

    def val_dataset(self, mode="val"):
        assert mode in ["val", "test"]
        if mode == "val":
            datasets = self.hparams.DATASET.VAL_DS.split("_")
        else:
            datasets = self.hparams.DATASET.TEST_DS.split("_")
        logger.info(f"Validation datasets are: {datasets}")
        val_datasets = []
        for dataset_name in datasets:
            ds = DatasetHMR(
                options=self.hparams.DATASET,
                dataset=dataset_name,
                is_train=False,
            )
            val_datasets.append({"ds": ds, "name": ds.dataset})

        return val_datasets

    def val_dataloader(self):
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds["ds"],
                    batch_size=self.hparams.DATASET.EVAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                    drop_last=False,
                    persistent_workers=False,
                )
            )
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        bbox_scale = batch["scale"]
        batch_size = bbox_scale.shape[0]
        bbox_center = batch["center"]
        dataset_names = batch["dataset_name"]

        img_h = batch["orig_shape"][:, 0]
        img_w = batch["orig_shape"][:, 1]
        pred = self(
            batch,
            self.num_val_samples,
            bbox_center=bbox_center,
            bbox_scale=bbox_scale,
            img_w=img_w,
            img_h=img_h,
        )

        pred_vertices_all = pred["vertices"]
        gt_cam_vertices = batch["vertices"]
        reproj_all_kpts, pred_2d, gt_2d = self.reproj_metric(batch, pred)

        assert "3dpw" in dataset_names[0] or "emdb" in dataset_names[0]
        # Convert predicted vertices to SMPL Fromat
        # (6890, 10475) * (B, S, 10475, 3) => (B, S, 6890, 3)
        pred_vertices_all = torch.matmul(self.smplx2smpl, pred_vertices_all)
        if "3dpw" in dataset_names[0]:  # use 14j skeleton
            gt_keypoints_3d = torch.matmul(self.J_regressor, gt_cam_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, H36M_TO_J14, :]
            # (17, 6890) * (B, S, 6890, 3) => (B, S, 17, 3)
            pred_keypoints_3d = torch.matmul(self.J_regressor, pred_vertices_all)
            pred_pelvis = pred_keypoints_3d[:, :, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, :, H36M_TO_J14, :]
        else:  # "emdb" in dataset_names[0]: use 24 smpl joints
            gt_keypoints_3d = batch["smpl_joints24"]
            gt_pelvis = gt_keypoints_3d[:, [1, 2]].mean(dim=1, keepdims=True)
            # Get all 24 smpl joints
            pred_keypoints_3d = torch.matmul(
                self.smpl_neutral.J_regressor, pred_vertices_all
            )
            pred_pelvis = pred_keypoints_3d[:, :, [1, 2]].mean(dim=2, keepdims=True)
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        pred_verts_centered_all = pred_vertices_all - pred_pelvis
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
        gt_verts_centered = gt_cam_vertices - gt_pelvis

        # MPJPE
        error = (
            torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d.unsqueeze(1)) ** 2).sum(dim=-1))
            .mean(dim=-1)
            .cpu()
            .numpy()
            * 1000
        )
        # PVE
        verts_error = (
            torch.sqrt(
                ((pred_verts_centered_all - gt_verts_centered.unsqueeze(1)) ** 2).sum(dim=-1)
            ).mean(dim=-1)
        ).cpu().numpy() * 1000
        # PA-MPJPE
        pred_keypoints_3d = pred_keypoints_3d.reshape(batch_size * self.num_val_samples, -1, 3)
        gt_keypoints_3d = expand_annotation(gt_keypoints_3d, self.num_val_samples)
        gt_keypoints_3d = gt_keypoints_3d.cpu()
        # Procrustes alignment faster on cpu than gpu
        S1_hat = compute_similarity_transform_torch(pred_keypoints_3d.cpu(), gt_keypoints_3d)
        r_error = (
            torch.sqrt(((S1_hat - gt_keypoints_3d) ** 2).sum(dim=-1))
            .mean(dim=-1)
            .reshape(batch_size, self.num_val_samples)
            .numpy()
            * 1000
        )
        # distribution diversity metrics:
        # get 17 coco joints to compute diversity metrics:
        joints_3d_smpl = torch.matmul(self.smpl_neutral.J_regressor, pred_vertices_all)
        joints_3d_smpl = self.smpl_neutral.vertex_joint_selector(
            pred_vertices_all.reshape(-1, 6890, 3), joints_3d_smpl.reshape(-1, 24, 3)
        ).reshape(batch_size, self.num_val_samples, 45, 3)
        joints_3d_coco = joints_3d_smpl[:, :, SMPL45_TO_COCO25][:, :, COCO25_TO_COCO17]

        joints_3d_sample_mean = joints_3d_coco.mean(dim=1).unsqueeze(1)
        joints_3d_diversity = (
            torch.linalg.vector_norm(joints_3d_coco - joints_3d_sample_mean, dim=-1)
            .cpu()
            .numpy()
            * 1000
        )

        joint_vis = batch["gtkps_coco"][:, :, 2] > self.hparams.DATASET.CONF_VIS_TH
        joint_vis = joint_vis.unsqueeze(1).expand(-1, self.num_val_samples, -1).cpu().numpy()
        num_vis_joints = joint_vis.sum()
        num_invis_joints = (~joint_vis).sum()
        # 3d joint diversity for vis/invis joints
        joints_3d_diversity_vis = (joints_3d_diversity * joint_vis).sum() / (
            num_vis_joints + 1e-6
        )
        joints_3d_diversity_invis = (joints_3d_diversity * ~joint_vis).sum() / (
            num_invis_joints + 1e-6
        )

        # 2d image consistency:
        n_vis_j_samples = joint_vis[:, 1:].sum()
        n_vis_j_mode = joint_vis[:, 0].sum()
        loss_2d_vis = (reproj_all_kpts[:, 1:] * joint_vis[:, 1:]).sum() / (
            n_vis_j_samples + 1e-6
        )
        loss_2d_vis_mode = (reproj_all_kpts[:, 0] * joint_vis[:, 0]).sum() / (
            n_vis_j_mode + 1e-6
        )
        # errors are of shape (B, S)
        loss_dict = {}
        ds_name = self.val_ds[dataloader_idx]["name"]
        # compute mask metrics only for subset of emdb-p1
        if dataset_names[0] == "emdb-p1-occl":
            mask_indices = batch["person_mask_indices"]
            mask_label = batch["person_mask_label"].cpu().numpy()
            # use mask and pred 2d kpts in cropped 256x256
            # for each invis joint, compute for each hypo the min distance to mask
            joints_2d = torch.from_numpy(pred_2d.reshape(batch_size, -1, 2)).to("cuda")
            distances = torch.cdist(joints_2d, mask_indices, p=2)
            values, _ = torch.min(distances, dim=-1)  # (bs, num_hypo * num_joints)
            # distances cannot be zero because gt indices are integers
            # min distance is ~0.71, 0.71 = sqrt(0.5² + 0.5²)
            values[values < 0.71] = 0
            # only compute metric for invisible joints and when mask is valid
            used_joints = (~joint_vis) * mask_label[:, None, None]
            values = values.reshape(batch_size, self.num_val_samples, 17).cpu().numpy()
            inside_ratio = (values == 0)
            # set to 0 for invalid masks and correctly save average of both metrics
            n_used_joints = used_joints.sum()
            mask_min_dist = (values * used_joints).sum() / (
                n_used_joints + 1e-6
            )
            mask_perc_in = (inside_ratio * used_joints).sum() / (
                n_used_joints + 1e-6
            )
            loss_dict[ds_name + "_mask-shortest-dist"] = (mask_min_dist, n_used_joints)
            loss_dict[ds_name + "_inside-mask-bool"] = (mask_perc_in, n_used_joints)
        else:
            # accuracy of predicted distribution
            loss_dict[ds_name + "_mpjpe-mode"] = error[:, 0]
            loss_dict[ds_name + "_pampjpe-mode"] = r_error[:, 0]
            loss_dict[ds_name + "_pve-mode"] = verts_error[:, 0]
            loss_dict[ds_name + "_mpjpe-min"] = error.min(axis=1)
            loss_dict[ds_name + "_pve-min"] = verts_error.min(axis=1)
            loss_dict[ds_name + "_pampjpe-min"] = r_error.min(axis=1)
            # sample-input consistency
            loss_dict[ds_name + "_2d-reproj-vis"] = (loss_2d_vis, n_vis_j_samples)
            loss_dict[ds_name + "_2d-reproj-vis-mode"] = (loss_2d_vis_mode, n_vis_j_mode)
            # distribution diversity
            # keep track of num_vis_joints to weigh metric correctly (on_validation_epoch_end)
            loss_dict[ds_name + "_3d-joint-diversity-vis"] = (
                joints_3d_diversity_vis,
                num_vis_joints,
            )
            loss_dict[ds_name + "_3d-joint-diversity-invis"] = (
                joints_3d_diversity_invis,
                num_invis_joints,
            )

        self.validation_step_outputs.append(loss_dict)

        # visualize a few randomly selected samples:
        if batch_nb in self.val_viz_batch_idx[dataloader_idx]:
            batch_idx = np.random.randint(low=0, high=batch_size)
            vertices_samples = pred_vertices_all[batch_idx]
            if vertices_samples.shape[1] == 10475:
                # always visualize smpl vertices for simplicity
                vertices_samples = torch.matmul(self.smplx2smpl, vertices_samples)
            pred_cam_t = pred["pred_cam_t"][batch_idx].float()

            batch["cam_intrinsics"] = pred["cam_intrinsics"][batch_idx]
            vertices2d, np_img, out_filename = self.prepare_viz_data(
                batch, vertices_samples, pred_cam_t, batch_idx
            )
            out_filename += "_" + str(batch_nb)  # to get unique filename
            directional_vertex_stddev, vertex_uncertainty_colours = vertex_diversity_from_samples(
                vertices_samples
            )

            # transform 2d vertices from full img to crop img
            center = bbox_center[batch_idx][None, :].cpu().numpy()
            scale = bbox_scale[batch_idx].cpu().numpy()
            vertices2d[0] = (vertices2d[0] - center) / (scale * 200)
            vertices2d[0] = (vertices2d[0] + 0.5) * self.hparams.DATASET.IMG_RES

            viz_mesh_xyz_std_dev(
                np_img,
                batch["imgname"][batch_idx],
                vertices2d[0],  # select mode
                directional_vertex_stddev.cpu().numpy(),
                num_samples=vertices2d.shape[0],
                out_filename=out_filename,
            )
            log_prob = pred["log_prob"]
            imgname = batch["imgname"][batch_idx]
            cv_img = read_img(imgname)
            if "closeup" in batch["dataset_name"][batch_idx]:
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

            focal_length = pred["cam_intrinsics"][batch_idx][[0, 1], [0, 1]]

            viz_3d_samples(
                vertices_samples,
                pred_cam_t,
                focal_length,
                cv_img,
                vertex_uncertainty_colours,
                out_filename,
                log_prob[batch_idx],
                center.squeeze(),
                scale.squeeze(),
                verts_error[batch_idx],
                self.hparams.DATASET.IMG_RES,
            )
            pred_2d = pred_2d[batch_idx] * (self.hparams.DATASET.IMG_RES / 256)
            gt_2d = gt_2d[batch_idx].squeeze() * (self.hparams.DATASET.IMG_RES / 256)
            nll = log_prob[batch_idx].cpu().numpy()
            plot_multiple_hypos(np_img, pred_2d, gt_2d, nll, out_filename, verts_error[batch_idx])
            scatter_joint_hypos(np_img, pred_2d, gt_2d, nll, out_filename)

    def prepare_viz_data(self, input_bach, vertices_samples, pred_cam_t, batch_idx):
        np_img = img_torch_to_np_denormalize(input_bach["img"][batch_idx].unsqueeze(0)).squeeze()
        dataset_name = input_bach["dataset_name"][batch_idx]

        save_dir = os.path.join(self.hparams.LOG_DIR, "output")
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f"{self.global_step:08d}_{dataset_name}")

        vertices2d = (
            perspective_projection(
                vertices_samples,
                translation=pred_cam_t,
                cam_intrinsics=input_bach["cam_intrinsics"][None],
            )
            .cpu()
            .numpy()
        )
        return vertices2d, np_img, save_filename

    def reproj_metric(self, gt_batch, out_batch):
        # compute 2d reproj error:
        batch_size = out_batch["vertices"].shape[0]
        num_samples = out_batch["vertices"].shape[1]

        pred_vertices_smplx = out_batch["vertices"]
        # need to first generate SMPL joints from SMPL-X mesh
        pred_vertices_smpl = torch.matmul(self.smplx2smpl, pred_vertices_smplx)
        # transform to smpl joints
        joints_3d_smpl = torch.matmul(self.smpl_neutral.J_regressor, pred_vertices_smpl)
        # get COCO joints
        joints_3d_smpl = self.smpl_neutral.vertex_joint_selector(
            pred_vertices_smpl.reshape(-1, 6890, 3), joints_3d_smpl.reshape(-1, 24, 3)
        ).reshape(batch_size, num_samples, 45, 3)
        joints_3d_smpl = joints_3d_smpl[:, :, SMPL45_TO_COCO25][:, :, COCO25_TO_COCO17]

        # project to 2d
        joints_3d_smpl = joints_3d_smpl.reshape(batch_size * num_samples, -1, 3)
        transl = out_batch["pred_cam_t"].reshape(batch_size * num_samples, 3)
        cam_intrinsics = expand_annotation(out_batch["cam_intrinsics"], num_samples)

        pred_keypoints_2d = perspective_projection(
            joints_3d_smpl,
            translation=transl,
            cam_intrinsics=cam_intrinsics,
        ).reshape(batch_size, num_samples, -1, 2)

        # normalize to [-1, 1]
        # Use full image keypoints
        gt_keypoints_2d = gt_batch["gtkps_coco_orig"].clone()
        center = gt_batch["center"][:, None]
        scale = gt_batch["scale"][:, None, None] * 200
        # normalization to cropped image [-1, 1]
        gt_keypoints_2d[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] - center) / scale
        pred_keypoints_2d[:, :, :, :2] = (
            2 * (pred_keypoints_2d[:, :, :, :2] - center[:, None]) / scale[:, None]
        )

        # scale to 256x256 for uniform 2d distance computation
        pred_keypoints_2d = (pred_keypoints_2d + 1.0) * (256 / 2.0)
        gt_keypoints_2d = (gt_keypoints_2d[:, :, :2] + 1.0) * (256 / 2.0)
        gt_keypoints_2d = gt_keypoints_2d.unsqueeze(1)
        eucl_dist_all = torch.linalg.vector_norm(pred_keypoints_2d - gt_keypoints_2d, dim=-1)
        return (
            eucl_dist_all.cpu().numpy(),
            pred_keypoints_2d.cpu().numpy(),
            gt_keypoints_2d.cpu().numpy(),
        )
