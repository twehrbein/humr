import argparse
import time
import os
import torch
from torch.utils.data import DataLoader
import cv2
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO
from loguru import logger

from easy_vitpose.model import ViTPose
from easy_vitpose.util import dyn_model_import
from easy_vitpose.top_down_eval import keypoints_from_heatmaps
from train.models.humr_model import Humr
from train.dataset.demo_dataset import DetectDataset
from train.core.config import update_hparams, SMPLX2SMPL, SMPL_MODEL_DIR
from train.utils.image_utils import img_torch_to_np_denormalize
from train.dataset.dataset import j2d_processing
from train.models.head.smpl_head import SMPL
from train.utils.vis_utils import (
    viz_mesh_xyz_std_dev,
    scatter_joint_hypos,
    viz_3d_samples,
)
from train.core.constants import (
    SMPL45_TO_COCO25,
    COCO25_TO_COCO17,
)
from train.utils.sampling_utils import vertex_diversity_from_samples, expand_annotation
from train.utils.geometry import perspective_projection


def demo(hparams):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    smpl_neutral = SMPL(SMPL_MODEL_DIR, gender="neutral", create_transl=False).to(DEVICE)
    # load humr model
    humr_model = Humr(
        backbone=hparams.MODEL.BACKBONE,
        img_res=hparams.DATASET.IMG_RES,
        pretrained_ckpt=hparams.TRAINING.PRETRAINED_CKPT,
        hparams=hparams,
    ).to(DEVICE)
    weights = torch.load(hparams.TRAINING.RESUME_CKPT, map_location=DEVICE)["state_dict"]
    # rename keys:
    state_dict = {}
    for key, value in weights.items():
        state_dict[key.replace("model.", "")] = value
    humr_model.load_state_dict(state_dict, strict=False)
    humr_model.eval()
    logger.info("humr model loaded")

    # load 2d detector
    model_name = "h"
    model_path = f"data/ckpt/vitpose-{model_name}-multi-coco.pth"
    data_type = "coco"
    model_cfg = dyn_model_import(data_type, model_name)
    pose2d_model = ViTPose(model_cfg).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    if "state_dict" in ckpt:
        pose2d_model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        pose2d_model.load_state_dict(ckpt, strict=True)
    pose2d_model.eval()
    logger.info("2D detector loaded")

    # Load a COCO-pretrained YOLO11 model for bounding box detection
    # smallest: yolo11n.pt, s, m, l, x
    yolo_model = YOLO("yolo11l.pt", task="detect")

    smplx2smpl = pickle.load(open(SMPLX2SMPL, "rb"))
    smplx2smpl = torch.tensor(smplx2smpl["matrix"], dtype=torch.float32).to("cuda")
    num_samples = hparams.TRAINING.NUM_TEST_SAMPLES
    # process img folder:
    imgfiles = sorted(glob("data/examples/*"))
    print(imgfiles)
    for imgpath in tqdm(imgfiles):
        imgname = Path(imgpath).stem
        # --- Detection ---
        with torch.no_grad():
            results = yolo_model(imgpath, classes=[0], conf=0.5, verbose=False)[0]
        boxes = np.array([r[:5].tolist() for r in results.boxes.data.cpu().numpy()]).reshape(
            (-1, 5)
        )

        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        full_img = img.copy()

        # create dataset with current image
        db = DetectDataset(img, boxes)
        dataloader = DataLoader(db, batch_size=8, shuffle=False, num_workers=1)

        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items() if type(v) is torch.Tensor}
            img = batch["img"]
            img2 = batch["img2"]
            batch_size = len(img)

            with torch.no_grad():
                heatmaps = pose2d_model(img2).cpu().numpy()

            center = batch["center"].cpu().numpy()
            scale = batch["scale"].cpu().numpy()
            scale_2d = np.repeat(scale[:, None], 2, axis=1) * 200
            scale_2d[:, 0] *= 192 / 256
            points, prob = keypoints_from_heatmaps(
                heatmaps=heatmaps,
                center=center,
                scale=scale_2d,
                unbiased=True,
                use_udp=True,
            )
            # pred in original image space
            pred_2d = np.concatenate([points, prob], axis=2)

            cond_kpts = []
            for i in range(batch_size):
                cond_kpts.append(
                    j2d_processing(pred_2d[i], center[i], scale[i], hparams.DATASET.IMG_RES)
                )
            cond_kpts = np.stack(cond_kpts, axis=0)
            cond_kpts = torch.from_numpy(cond_kpts).float().to(DEVICE)
            batch["cond_keypoints"] = cond_kpts

            np_img = img_torch_to_np_denormalize(img)
            img_h = batch["orig_shape"][:, 0]
            img_w = batch["orig_shape"][:, 1]
            with torch.no_grad():
                pred = humr_model(
                    batch,
                    num_samples,
                    bbox_center=batch["center"],
                    bbox_scale=batch["scale"],
                    img_w=img_w,
                    img_h=img_h,
                    fl=None,
                )

            # VISUALIZE PREDICTIONS:
            pred_vertices_all = pred["vertices"]
            # convert to SMPL vertices
            pred_vertices_all = torch.matmul(smplx2smpl, pred_vertices_all)
            pred_cam_t = pred["pred_cam_t"].float()

            # compute 2d joint reprojections:
            joints_3d_smpl = torch.matmul(smpl_neutral.J_regressor, pred_vertices_all)
            # get COCO joints
            joints_3d_smpl = smpl_neutral.vertex_joint_selector(
                pred_vertices_all.reshape(-1, 6890, 3), joints_3d_smpl.reshape(-1, 24, 3)
            ).reshape(batch_size, num_samples, 45, 3)
            joints_3d_coco = joints_3d_smpl[:, :, SMPL45_TO_COCO25][:, :, COCO25_TO_COCO17]
            # project to 2d
            joints_3d_coco = joints_3d_coco.reshape(batch_size * num_samples, -1, 3)
            joint_hypos_2d = (
                perspective_projection(
                    joints_3d_coco,
                    translation=pred_cam_t.reshape(batch_size * num_samples, 3),
                    cam_intrinsics=expand_annotation(pred["cam_intrinsics"], num_samples),
                )
                .reshape(batch_size, num_samples, -1, 2)
                .cpu()
                .numpy()
            )
            # normalize to [0, crop_res]
            center_ = center[:, None, None]
            scale_ = scale[:, None, None, None] * 200
            joint_hypos_2d[:, :, :, :2] = 2 * (joint_hypos_2d[:, :, :, :2] - center_) / scale_
            joint_hypos_2d = (joint_hypos_2d + 1.0) * (hparams.DATASET.IMG_RES / 2.0)

            # visualize mode prediction:
            vertices2d = (
                (
                    perspective_projection(
                        pred_vertices_all[:, 0],
                        translation=pred_cam_t[:, 0],
                        cam_intrinsics=pred["cam_intrinsics"],
                    )
                )
                .cpu()
                .numpy()
            )  # (bs, 6890, 2)
            # transform 2d vertices from full img to crop img
            vertices2d = (vertices2d - center[:, None]) / (scale[:, None, None] * 200)
            vertices2d = (vertices2d + 0.5) * hparams.DATASET.IMG_RES
            # transform vitpose 2d detections to crop space
            kpts_crop = cond_kpts.cpu().numpy()
            kpts_crop = (kpts_crop + 1.0) * 224 * 0.5

            for i in range(batch_size):
                out_filename = os.path.join(hparams.LOG_DIR, f"{imgname}_{i}")
                directional_vertex_stddev, vertex_uncertain_colours = vertex_diversity_from_samples(
                    pred_vertices_all[i]
                )
                viz_mesh_xyz_std_dev(
                    np_img[i],
                    imgpath,
                    vertices2d[i],
                    directional_vertex_stddev.cpu().numpy(),
                    num_samples=num_samples,
                    out_filename=out_filename,
                )
                viz_3d_samples(
                    pred_vertices_all[i],
                    pred_cam_t[i],
                    pred["cam_intrinsics"][i][[0, 1], [0, 1]],
                    full_img,
                    vertex_uncertain_colours,
                    out_filename,
                    pred["log_prob"][i],
                    center[i].squeeze(),
                    scale[i].squeeze(),
                    None,
                    hparams.DATASET.IMG_RES,
                )
                nll = pred["log_prob"][i].cpu().numpy()
                scatter_joint_hypos(np_img[i], joint_hypos_2d[i], kpts_crop[i], nll, out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="cfg file path", required=True)
    parser.add_argument("--log_dir", type=str, help="log dir path", default="./logs")
    parser.add_argument("--ckpt", type=str, help="path to checkpoint to load", required=True)
    parser.add_argument("--name", help="name of experiment", default="demo", type=str)

    args = parser.parse_args()
    hparams = update_hparams(args.cfg)
    hparams.EXP_NAME = args.name
    hparams.TRAINING.RESUME_CKPT = args.ckpt

    # add date to log path
    logtime = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(args.log_dir, logtime + "_" + hparams.EXP_NAME)
    os.makedirs(logdir, exist_ok=False)
    hparams.LOG_DIR = logdir
    demo(hparams)
