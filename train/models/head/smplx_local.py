from smplx import SMPLXLayer as SMPLX_
from smplx.utils import SMPLXOutput
from pytorch3d.transforms import axis_angle_to_matrix

from ...core.constants import NUM_JOINTS_SMPLX


class SMPLX(SMPLX_):
    """Input per default is rotation matrices.
    flat_hand_mean is always True?!

    Args:
        SMPLX_ (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)

    def forward(self, pose2rot=False, *args, **kwargs):
        if pose2rot:
            body_pose = axis_angle_to_matrix(kwargs["body_pose"].reshape(-1, 3)).reshape(
                -1, NUM_JOINTS_SMPLX - 1, 3, 3
            )
            global_orient = axis_angle_to_matrix(kwargs["global_orient"].reshape(-1, 3)).reshape(
                -1, 1, 3, 3
            )
            kwargs["body_pose"] = body_pose
            kwargs["global_orient"] = global_orient

        smplx_output = super(SMPLX, self).forward(*args, **kwargs)
        output = SMPLXOutput(
            vertices=smplx_output.vertices,
            global_orient=smplx_output.global_orient,
            body_pose=smplx_output.body_pose,
            joints=smplx_output.joints,
            betas=smplx_output.betas,
            full_pose=smplx_output.full_pose,
            transl=smplx_output.transl,
        )
        return output
