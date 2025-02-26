import torch.nn as nn

from .smplx_local import SMPLX
from ...core import config
from ...utils.geometry import perspective_projection, convert_pare_to_full_img_cam


class SMPLXCamHead(nn.Module):
    def __init__(self, img_res):
        super(SMPLXCamHead, self).__init__()
        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)
        self.img_res = img_res

    def forward(self, rotmat, shape, cam, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h):
        smpl_output = self.smplx(
            betas=shape,
            body_pose=rotmat[:, 1:22].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        )

        output = {
            "vertices": smpl_output.vertices,
            "joints3d": smpl_output.joints,
        }

        joints3d = output["joints3d"]

        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.0,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
        )

        joints2d = perspective_projection(
            joints3d,
            cam_intrinsics=cam_intrinsics,
            translation=cam_t,
        )

        output["joints2d"] = joints2d
        output["pred_cam_t"] = cam_t

        return output
