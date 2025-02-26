# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

NUM_JOINTS_SMPLX = 22  # including root
NUM_JOINTS_SMPL = 24  # including root
NUM_BETAS = 11

COCO17_BONES = [[0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [5, 6],
                [5, 7],
                [7, 9],
                [6, 8],
                [8, 10],
                [5, 11],
                [6, 12],
                [11, 12],
                [11, 13],
                [13, 15],
                [12, 14],
                [14, 16]]

# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py#L97
SMPL45_TO_COCO25 = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26,
                    27, 28, 29, 30, 31, 32, 33, 34]
# ordering accoring to coco17 skeleton
COCO25_TO_COCO17 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py#L117
SMPLX_TO_COCO25 = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57,
                   58, 59, 60, 61, 62, 63, 64, 65]

"""
We create a superset of joints containing the OpenPose joints together
with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    "OP Nose",  # 0
    "OP Neck",  # 1
    "OP RShoulder",  # 2
    "OP RElbow",  # 3
    "OP RWrist",  # 4
    "OP LShoulder",  # 5
    "OP LElbow",  # 6
    "OP LWrist",  # 7
    "OP MidHip",  # 8
    "OP RHip",  # 9
    "OP RKnee",  # 10
    "OP RAnkle",  # 11
    "OP LHip",  # 12
    "OP LKnee",  # 13
    "OP LAnkle",  # 14
    "OP REye",  # 15
    "OP LEye",  # 16
    "OP REar",  # 17
    "OP LEar",  # 18
    "OP LBigToe",  # 19
    "OP LSmallToe",  # 20
    "OP LHeel",  # 21
    "OP RBigToe",  # 22
    "OP RSmallToe",  # 23
    "OP RHeel",  # 24
    # 24 Ground Truth joints (superset of joints from different datasets)
    "Right Ankle",  # 0
    "Right Knee",  # 1
    "Right Hip",  # 2
    "Left Hip",  # 3
    "Left Knee",  # 4
    "Left Ankle",  # 5
    "Right Wrist",  # 6
    "Right Elbow",  # 7
    "Right Shoulder",  # 8
    "Left Shoulder",  # 9
    "Left Elbow",  # 10
    "Left Wrist",  # 11
    "Neck (LSP)",  # 12
    "Top of Head (LSP)",  # 13
    "Pelvis (MPII)",  # 14
    "Thorax (MPII)",  # 15
    "Spine (H36M)",  # 16
    "Jaw (H36M)",  # 17
    "Head (H36M)",  # 18
    "Nose",  # 19
    "Left Eye",  # 20
    "Right Eye",  # 21
    "Left Ear",  # 22
    "Right Ear",  # 23
]

# Map joints to SMPL joints
JOINT_MAP = {
    "OP Nose": 24,
    "OP Neck": 12,
    "OP RShoulder": 17,
    "OP RElbow": 19,
    "OP RWrist": 21,
    "OP LShoulder": 16,
    "OP LElbow": 18,
    "OP LWrist": 20,
    "OP MidHip": 0,
    "OP RHip": 2,
    "OP RKnee": 5,
    "OP RAnkle": 8,
    "OP LHip": 1,
    "OP LKnee": 4,
    "OP LAnkle": 7,
    "OP REye": 25,
    "OP LEye": 26,
    "OP REar": 27,
    "OP LEar": 28,
    "OP LBigToe": 29,
    "OP LSmallToe": 30,
    "OP LHeel": 31,
    "OP RBigToe": 32,
    "OP RSmallToe": 33,
    "OP RHeel": 34,
    "Right Ankle": 8,
    "Right Knee": 5,
    "Right Hip": 45,
    "Left Hip": 46,
    "Left Knee": 4,
    "Left Ankle": 7,
    "Right Wrist": 21,
    "Right Elbow": 19,
    "Right Shoulder": 17,
    "Left Shoulder": 16,
    "Left Elbow": 18,
    "Left Wrist": 20,
    "Neck (LSP)": 47,
    "Top of Head (LSP)": 48,
    "Pelvis (MPII)": 49,
    "Thorax (MPII)": 50,
    "Spine (H36M)": 51,
    "Jaw (H36M)": 52,
    "Head (H36M)": 53,
    "Nose": 24,
    "Left Eye": 26,
    "Right Eye": 25,
    "Left Ear": 28,
    "Right Ear": 27,
}

COCO_17 = {
    0: "Nose",
    1: "eye_l",
    2: "eye_r",
    3: "ear_l",
    4: "ear_r",
    5: "shoulder_l",
    6: "shoulder_r",
    7: "elbow_l",
    8: "elbow_r",
    9: "wrist_l",
    10: "wrist_r",
    11: "hip_l",
    12: "hip_r",
    13: "knee_l",
    14: "knee_r",
    15: "ankle_l",
    16: "ankle_r",
}
