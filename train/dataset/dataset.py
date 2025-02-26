import cv2
import os
import torch
import numpy as np
from loguru import logger
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import gzip
from smplx import SMPL

from ..core.constants import (
    NUM_JOINTS_SMPLX,
    COCO25_TO_COCO17,
    IMG_NORM_MEAN,
    IMG_NORM_STD,
)
from ..core.config import (
    DATASET_FILES,
    DATASET_FOLDERS,
    ALL_BEDLAM_SETS,
    ENV_MASK_PATH,
    HM_SAMPLES_PATH,
    SMPL_MODEL_DIR,
    SMPLX_MODEL_DIR,
)
from ..utils.image_utils import (
    crop,
    random_crop,
    read_img,
    get_transform,
    transform,
)
from train.models.head.smplx_local import SMPLX


class DatasetHMR(Dataset):
    def __init__(self, options, dataset, is_train=True):
        super(DatasetHMR, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.hm_samples_path = os.path.join(HM_SAMPLES_PATH, self.dataset)
        annot_path = DATASET_FILES[is_train][dataset]
        with np.load(annot_path, allow_pickle=True) as annot_data:
            data = dict(annot_data)
        assert "vitpose" in data
        vitpose_detections = data["vitpose"][:, :, [1, 0, 2]]

        confidences = np.clip(vitpose_detections[:, :, -1], 0, 1)
        vitpose_detections[:, :, -1] = confidences
        assert vitpose_detections.shape[2] == 3
        assert vitpose_detections.shape[0] == len(data["imgname"])
        self.hm_samples_indices = np.arange(len(data["imgname"]))

        self.gtkps_coco = data["gt_coco_25j"][:, COCO25_TO_COCO17]
        conf = vitpose_detections[:, :, [-1]]
        self.gtkps_coco = np.concatenate((self.gtkps_coco, conf), axis=-1)

        self.cond_keypoints = vitpose_detections
        # filter out examples with bad 2d keypoint predictions:
        eucl_dist = np.linalg.norm(
            self.gtkps_coco[:, :, :2] - self.cond_keypoints[:, :, :2], axis=-1
        )
        valid_joints = self.cond_keypoints[:, :, 2] > 0.2
        box_scale = data["scale"].astype(np.float32)
        valid_eucl_dist = (eucl_dist * valid_joints).mean(axis=-1) / box_scale
        err_th = 13
        valid_examples_err = valid_eucl_dist < err_th
        # filter out depending on number of vis joints
        vis_joints_02 = (self.cond_keypoints[:, :, 2] > 0.5).sum(axis=-1)
        valid_examples_vis = vis_joints_02 > 8
        valid_examples = np.logical_and(valid_examples_err, valid_examples_vis)
        # filter out samples with 2d predictions outside of image (outliers)
        if "3dpw" not in self.dataset:
            allowed_pad = 100
            if "closeup" in self.dataset:
                min_allowed_x = 0 - allowed_pad
                max_allowed_x = 720 + allowed_pad
                min_allowed_y = 0 - allowed_pad
                max_allowed_y = 1280 + allowed_pad
            else:
                min_allowed_x = 0 - allowed_pad
                max_allowed_x = 1280 + allowed_pad
                min_allowed_y = 0 - allowed_pad
                max_allowed_y = 720 + allowed_pad
            valid_pos = (
                (min_allowed_x <= self.cond_keypoints[:, :, 0])
                * (self.cond_keypoints[:, :, 0] <= max_allowed_x)
                * (min_allowed_y <= self.cond_keypoints[:, :, 1])
                * (self.cond_keypoints[:, :, 1] <= max_allowed_y)
            ).all(axis=-1)
            valid_examples = np.logical_and(valid_examples, valid_pos)

        # filter training datasets w.r.t. 2d detection quality
        if self.is_train and self.options.FILTER_DATASET:
            n_invalid = (~valid_examples).sum()
            perc = 100 * n_invalid / len(valid_examples)
            logger.info(f"Filtered {n_invalid} from {len(valid_examples)}, {perc:.2f}%")
        else:
            valid_examples = np.full(valid_examples.shape, True)

        self.gtkps_coco = self.gtkps_coco[valid_examples]
        self.cond_keypoints = self.cond_keypoints[valid_examples]

        if "agora" in self.dataset or self.dataset in ALL_BEDLAM_SETS:
            # load obj occlusion labels (denotes if for a person, an object occludes part of it)
            self.obj_occlusions = data["object_occlusion"][valid_examples]
        else:
            # don't have person masks for datasets other than agora or bedlam
            # set obj occlusion to true, such that these examples are ignored for mask loss
            self.obj_occlusions = np.ones((len(self.gtkps_coco)), dtype=bool)

        self.data_keys = list(data.keys())

        self.imgname = data["imgname"][valid_examples]
        self.hm_samples_indices = self.hm_samples_indices[valid_examples]
        self.scale = data["scale"][valid_examples].astype(np.float32)
        self.center = data["center"][valid_examples].astype(np.float32)

        if "cam_int" in data:
            self.cam_int = data["cam_int"][valid_examples]
        else:
            self.cam_int = np.zeros((self.imgname.shape[0], 3, 3), dtype=np.float32)
        if "orig_shape" in data:
            self.orig_shapes = data["orig_shape"][valid_examples]

        if self.is_train:
            self.pose_cam = data["pose_cam"][valid_examples][:, : NUM_JOINTS_SMPLX * 3].astype(
                np.float32
            )
            self.betas = data["shape"][valid_examples].astype(np.float32)
            self.keypoints = data["gtkps"][valid_examples][:, :24]
            if self.keypoints.shape[2] == 2:
                self.keypoints = np.concatenate(
                    (self.keypoints, np.ones_like(self.keypoints)[:, :, [0]]), axis=2
                )
        else:
            assert "3dpw" in self.dataset or "emdb" in self.dataset
            self.pose_cam = data["pose_cam"][valid_examples].astype(np.float32)
            self.betas = data["shape"][valid_examples].astype(np.float32)

        try:
            gender = data["gender"][valid_examples]
            for g in gender:
                assert g[0] == "m" or g[0] == "f"
            self.gender = np.array([0 if str(g[0]) == "m" else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        if self.dataset == "emdb-p1-occl":
            # load annot for which mask rcnn mask is okay/good enough
            # only use valid and fully inside img ones
            # only used for evaluation
            self.valid_emdb_mask_label = data["valid_emdb_mask"] == 1

        if not self.is_train and ("3dpw" in self.dataset or "emdb" in self.dataset):
            self.smpl_male = SMPL(SMPL_MODEL_DIR, gender="male", create_transl=False)
            self.smpl_female = SMPL(SMPL_MODEL_DIR, gender="female", create_transl=False)
        if not self.is_train and "bedlam" in self.dataset:
            self.smplx = SMPLX(SMPLX_MODEL_DIR, num_betas=11)

        self.length = self.scale.shape[0]
        # self.length = min(100, self.length)
        del data
        logger.info(f"Loaded {self.dataset} dataset, num samples {self.length}")

    def scale_aug(self):
        sc = 1.0
        if self.is_train:
            sc = min(
                1 + self.options.SCALE_FACTOR,
                max(
                    1 - self.options.SCALE_FACTOR, np.random.randn() * self.options.SCALE_FACTOR + 1
                ),
            )
        return sc

    def rgb_processing(self, rgb_img_full, center, scale, img_res):
        if self.is_train and self.options.ALB:
            aug_comp = [
                A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                A.ImageCompression(20, 100, p=0.1),
                A.RandomRain(blur_value=4, p=0.1),
                A.MotionBlur(blur_limit=(3, 15), p=0.2),
                A.Blur(blur_limit=(3, 10), p=0.1),
                A.RandomSnow(brightness_coeff=1.5, snow_point_lower=0.2, snow_point_upper=0.4),
            ]
            aug_mod = [
                A.CLAHE((1, 11), (10, 10), p=0.2),
                A.ToGray(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.MultiplicativeNoise(
                    multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    always_apply=False,
                    p=0.2,
                ),
                A.Posterize(p=0.1),
                A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                A.Equalize(mode="cv", p=0.1),
            ]
            albumentation_aug = A.Compose(
                [
                    A.OneOf(aug_comp, p=self.options.ALB_PROB),
                    A.OneOf(aug_mod, p=self.options.ALB_PROB),
                ]
            )
            rgb_img_full = albumentation_aug(image=rgb_img_full)["image"]

        rgb_img = crop(rgb_img_full, center, scale, [img_res, img_res])
        rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
        return rgb_img

    def validate_crop(self, index, center, scale, sc, orig_shape, eps=10):
        # need to check if center and sc*scale produce valid crops
        # otherwise, choose default center and scale
        res = [self.options.IMG_RES, self.options.IMG_RES]
        ul = np.array(transform([1, 1], center, sc * scale, res, invert=1)) - 1
        br = np.array(transform([res[0] + 1, res[1] + 1], center, sc * scale, res, invert=1)) - 1
        img_h, img_w = orig_shape
        # crop is valid if upper left (ul) or bottom right (br) point is inside the original image
        if ul[1] + eps > img_h or br[1] - eps < 0 or ul[0] + eps > img_w or br[0] - eps < 0:
            logger.info("IMG CROP OUT OF IMG SPACE! reset scale and center to default")
            center = self.center[index].copy()
            scale = self.scale[index].copy()
            sc = 1
        return center, scale, sc

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        sc = self.scale_aug()

        # apply crop augmentation
        if self.is_train and self.options.CROP_FACTOR > 0:
            rand_no = np.random.rand()
            if rand_no < self.options.CROP_PROB:
                crop_f = self.options.CROP_FACTOR
                center, scale = random_crop(center, scale, crop_scale_factor=1 - crop_f, axis="y")
        if self.is_train:
            if "orig_shape" in self.data_keys:
                orig_shape = self.orig_shapes[index]
            else:
                if "closeup" in self.dataset:
                    orig_shape = np.array([1280, 720])
                else:
                    orig_shape = np.array([720, 1280])
            center, scale, sc = self.validate_crop(index, center, scale, sc, orig_shape)

        imgname = os.path.join(self.img_dir, self.imgname[index])
        if not os.path.isfile(imgname):
            imgname = imgname[:-4] + ".jpg"
        try:
            cv_img = read_img(imgname)
        except Exception as E:
            print(E, flush=True)
            logger.info(f"@{imgname}@ from {self.dataset}")
        if "closeup" in self.dataset:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
        if self.is_train:
            assert np.array_equal(orig_shape, np.array(cv_img.shape)[:2])
        else:
            orig_shape = np.array(cv_img.shape)[:2]
        # Process image
        try:
            orig_c = self.center[index].copy()
            orig_s = self.scale[index].copy()
            img = self.rgb_processing(cv_img, center, sc * scale, img_res=self.options.IMG_RES)
        except Exception as E:
            logger.info(f"@{imgname} from {self.dataset}")
            print(cv_img.shape, center, sc, scale, self.options.IMG_RES)
            print("Error message:", E, flush=True)

        img = torch.from_numpy(img).float()
        item["img"] = self.normalize_img(img)
        item["imgname"] = imgname

        # load env mask:
        # compute valid pixel position and save it in padded tensor with fixed length
        # need fixed length to compute batch-wise loss
        fix_len = int(self.options.IMG_RES * self.options.IMG_RES)
        if self.obj_occlusions[index]:
            indices = torch.full((fix_len, 2), -1, dtype=torch.float32)
            env_mask = torch.zeros(
                (self.options.IMG_RES, self.options.IMG_RES), dtype=torch.bool
            )
        else:
            dataset_name = imgname.split("/")[-4]
            seq_name = imgname.split("/")[-2]
            img_name = imgname.split("/")[-1][:-4]
            if "agora" in self.dataset:
                env_mask_name = img_name + "_all_person.png"
                mask_path = os.path.join(ENV_MASK_PATH, seq_name, env_mask_name)
            else:
                env_mask_name = img_name + "_env.png"
                mask_path = os.path.join(ENV_MASK_PATH, dataset_name, seq_name, env_mask_name)
            env_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            iterations = 3
            if "agora" in self.dataset:
                # person pixel should be 0, others 255
                # agora masks background pixel are 76?!
                bg_mask = env_mask == 76
                new_mask = np.zeros((env_mask.shape), dtype=np.uint8)
                new_mask[bg_mask] = 255
                env_mask = new_mask
            if "closeup" in self.dataset:
                env_mask = cv2.rotate(env_mask, cv2.ROTATE_90_CLOCKWISE)

            env_mask = crop(
                env_mask, center, sc * scale, [self.options.IMG_RES, self.options.IMG_RES]
            )
            env_mask = (env_mask < 255 / 2).astype(np.float32)
            # apply dilation to increase size of person masks a bit
            env_mask = cv2.dilate(env_mask, np.ones((3, 3), np.float32), iterations=iterations)
            env_mask = torch.from_numpy(env_mask > 0)

            # compute valid indices instead:
            indices = env_mask.nonzero().float()
            if len(indices) < 1:
                indices = torch.full((fix_len, 2), -1, dtype=torch.float32)
            else:
                # pad indices to fixed len
                padd = torch.tensor([[indices[0, 0], indices[0, 1]]], dtype=torch.float32)
                padd = padd.expand(fix_len - len(indices), -1)
                indices = torch.cat((indices, padd), dim=0)
                # need to swap columns
                h_indices = indices[:, 0].clone()
                indices[:, 0] = indices[:, 1]
                indices[:, 1] = h_indices
        item["valid_mask"] = indices[0, 0] >= 0
        item["mask_indices"] = indices
        item["env_mask"] = env_mask

        item["pose"] = torch.from_numpy(self.pose_cam[index].copy()).float()
        item["betas"] = torch.from_numpy(self.betas[index].copy()).float()

        if "cam_int" in self.data_keys:
            item["focal_length"] = torch.tensor(
                [self.cam_int[index][0, 0], self.cam_int[index][1, 1]]
            )

        gtkps_coco_orig = self.gtkps_coco[index].copy()
        gtkps_coco = j2d_processing(
            gtkps_coco_orig.copy(),
            center,
            sc * scale,
            self.options.IMG_RES
        )
        gtkps_coco_orig[:, 2] = gtkps_coco[:, 2].copy()
        item["gtkps_coco_orig"] = torch.from_numpy(gtkps_coco_orig).float()
        item["gtkps_coco"] = torch.from_numpy(gtkps_coco).float()
        # compute MMD loss only for highly articulated joints
        valid_j = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.bool8)
        item["valid_sample_joints"] = valid_j
        # and only for joints that are inside the image crop
        gt_inside_x = np.logical_and(gtkps_coco[:, 0] >= -1, gtkps_coco[:, 0] <= 1)
        gt_inside_y = np.logical_and(gtkps_coco[:, 1] >= -1, gtkps_coco[:, 1] <= 1)
        gt_square_inside = np.logical_and(gt_inside_x, gt_inside_y)
        item["coco_inside_square_crop"] = gt_square_inside
        coco_gt_bbox = j2d_processing(
            self.gtkps_coco[index].copy(),
            orig_c,
            orig_s,
            self.options.IMG_RES
        )
        gt_inside_x = np.logical_and(coco_gt_bbox[:, 0] >= -0.75, coco_gt_bbox[:, 0] <= 0.75)
        gt_inside_y = np.logical_and(coco_gt_bbox[:, 1] >= -1, coco_gt_bbox[:, 1] <= 1)
        gt_bbox_inside = np.logical_and(gt_inside_x, gt_inside_y)
        item["coco_gt_bbox_inside"] = gt_bbox_inside

        cond_kpts_orig = self.cond_keypoints[index].copy()
        cond_kpts = j2d_processing(
            cond_kpts_orig.copy(),
            center,
            sc * scale,
            self.options.IMG_RES
        )
        item["cond_keypoints"] = torch.from_numpy(cond_kpts).float()
        if self.is_train:
            # used for reproj loss during training
            keypoints_orig = self.keypoints[index].copy()
            item["keypoints_orig"] = torch.from_numpy(keypoints_orig).float()

            # load precomputed vitpose heatmap samples:
            hm_index = self.hm_samples_indices[index]
            hm_path = os.path.join(self.hm_samples_path, f"{hm_index:07d}.npy.gz")
            with gzip.GzipFile(hm_path, "r") as f:
                hm_samples_data = np.load(f)
            hm_max_confs = np.max(hm_samples_data[:, :, 2], axis=0)
            item["hm_max_confs"] = hm_max_confs

            np.random.shuffle(hm_samples_data)
            hm_samples = hm_samples_data[:, :, :2]
            process_samples = 150
            hm_samples = hm_samples[:process_samples]
            samples_mask_dist = hm_samples_data[
                :process_samples, :, 3
            ]
            samples_outside_mask = samples_mask_dist > 0.71
            if self.obj_occlusions[index]:
                item["samples_outside_mask"] = np.ones_like(samples_outside_mask)
            else:
                item["samples_outside_mask"] = samples_outside_mask

            item["hm_samples_orig"] = torch.from_numpy(hm_samples).float()

        item["scale"] = np.float32(sc * scale)
        item["center"] = center.astype(np.float32)
        item["orig_shape"] = orig_shape
        item["dataset_name"] = self.dataset

        if not self.is_train and self.dataset == "emdb-p1-occl":
            # load person mask for mask loss evaluation during testing
            mask_path = imgname.replace("EMDB/", "EMDB/masks/")
            mask_path = mask_path.replace("/images/", "/masks/")
            mask_path = mask_path.replace(".jpg", ".png")
            person_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # same as for training, construct vector of valid pixel position
            # evaluate 2d reproj in 256x256
            fix_len = int(256 * 256)  # 921600
            person_mask = crop(person_mask, center, sc * scale, [256, 256])
            person_mask = (person_mask > 0).astype(np.float32)
            person_mask = torch.from_numpy(person_mask)
            # compute valid indices instead:
            indices = person_mask.nonzero().float()
            assert len(indices) >= 1
            # pad indices to fixed len
            padd = torch.tensor([[indices[0, 0], indices[0, 1]]], dtype=torch.float32)
            padd = padd.expand(fix_len - len(indices), -1)
            indices = torch.cat((indices, padd), dim=0)
            # need to swap columns
            h_indices = indices[:, 0].clone()
            indices[:, 0] = indices[:, 1]
            indices[:, 1] = h_indices

            item["person_mask"] = person_mask
            item["person_mask_indices"] = indices
            item["person_mask_label"] = self.valid_emdb_mask_label[index]
        if not self.is_train:
            if "3dpw" in self.dataset or "emdb" in self.dataset:
                global_orient = item["pose"].unsqueeze(0)[:, :3]
                body_pose = item["pose"].unsqueeze(0)[:, 3:]
                betas = item["betas"].unsqueeze(0)

                if self.gender[index] == 1:
                    gt_smpl_out = self.smpl_female(
                        global_orient=global_orient,
                        body_pose=body_pose,
                        betas=betas,
                    )
                else:
                    gt_smpl_out = self.smpl_male(
                        global_orient=global_orient,
                        body_pose=body_pose,
                        betas=betas,
                    )
                gt_vertices = gt_smpl_out.vertices
                gt_joints24 = gt_smpl_out.joints[0, :24]
                item["smpl_joints24"] = gt_joints24
                item["vertices"] = gt_vertices[0]
        return item

    def __len__(self):
        return self.length


def j2d_processing(kp, center, scale, img_res):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    transform_matrix = get_transform(
        center,
        scale,
        [img_res, img_res],
    )
    # need 3rd dim to be 1 for transformation
    confidence_values = kp[:, 2].copy()
    kp[:, 2] = 1
    kp = np.dot(kp, transform_matrix.T).astype(int) + 1
    kp = kp.astype("float32")
    kp[:, 2] = confidence_values

    # convert to normalized coordinates to [-1, 1]
    kp[:, :-1] = 2.0 * kp[:, :-1] / img_res - 1.0
    return kp
