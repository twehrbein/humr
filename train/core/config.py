from yacs.config import CfgNode as CN
from os.path import join


SMPL_MODEL_DIR = "data/body_models/SMPL_python_v.1.1.0/smpl/models"
SMPLX_MODEL_DIR = "data/body_models/smplx/models/smplx"

JOINT_REGRESSOR_TRAIN_EXTRA = "data/utils/J_regressor_extra.npy"
JOINT_REGRESSOR_H36M = "data/utils/J_regressor_h36m.npy"
SMPL_MEAN_PARAMS = "data/utils/smpl_mean_params.npz"
SMPLX2SMPL = "data/utils/smplx2smpl.pkl"

# all bedlam datasets (without AGORA)
ALL_BEDLAM_SETS = ["static-office-hair", "zoom-suburbd-hair", "static-gym-hair", "static-hdri",
                   "zoom-suburbd", "closeup-suburba", "closeup-suburbb", "closeup-suburbc",
                   "closeup-suburbd", "closeup-gym", "zoom-gym", "static-gym", "static-office",
                   "orbit-office", "static-hdri-zoomed", "pitchup-stadium", "pitchdown-stadium",
                   "static-hdri-bmi", "closeup-suburbb-bmi", "closeup-suburbc-bmi",
                   "static-suburbd-bmi", "zoom-gym-bmi", "orbit-archviz-15", "orbit-archviz-19",
                   "orbit-archviz-12", "orbit-archviz-10", "orbit-stadium-bmi",
                   "orbit-archviz-objocc", "zoom-suburbb-frameocc", "static-hdri-frameocc"]

DATASET_FOLDERS = {
    "3dpw-test-cam": "data/test_images/3DPW",
    "emdb-p1": "data/test_images/EMDB",
    "emdb-p1-occl": "data/test_images/EMDB",
    "3dpw-train-smplx": "data/test_images/3DPW",
    # all training images used in bedlam-cliff/hmr 26 BEDLAM and 2 AGORA sequences:
    "agora-bfh": "data/training_images/images/",
    "agora-body": "data/training_images/images/",
    "zoom-suburbd": "data/training_images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png",
    "closeup-suburba": "data/training_images/20221011_1_250_batch01hand_closeup_suburb_a_6fps/png",
    "closeup-suburbb": "data/training_images/20221011_1_250_batch01hand_closeup_suburb_b_6fps/png",
    "closeup-suburbc": "data/training_images/20221011_1_250_batch01hand_closeup_suburb_c_6fps/png",
    "closeup-suburbd": "data/training_images/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png",
    "closeup-gym": "data/training_images/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps/png",
    "zoom-gym": "data/training_images/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps/png",
    "static-gym": "data/training_images/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps/png",
    "static-office": "data/training_images/20221013_3_250_batch01hand_static_bigOffice_6fps/png",
    "orbit-office": "data/training_images/20221013_3_250_batch01hand_orbit_bigOffice_6fps/png",
    "orbit-archviz-15": "data/training_images/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png",
    "orbit-archviz-19": "data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps/png",
    "orbit-archviz-12": "data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps/png",
    "orbit-archviz-10": "data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps/png",
    "static-hdri": "data/training_images/20221010_3_1000_batch01hand_6fps/png",
    "static-hdri-zoomed": "data/training_images/20221017_3_1000_batch01hand_6fps/png",
    "pitchup-stadium": "data/training_images/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps/png",
    "pitchdown-stadium": "data/training_images/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps/png",
    "static-hdri-bmi": "data/training_images/20221019_3_250_highbmihand_6fps/png",
    "closeup-suburbb-bmi": "data/training_images/20221019_1_250_highbmihand_closeup_suburb_b_6fps/png",
    "closeup-suburbc-bmi": "data/training_images/20221019_1_250_highbmihand_closeup_suburb_c_6fps/png",
    "static-suburbd-bmi": "data/training_images/20221019_3-8_1000_highbmihand_static_suburb_d_6fps/png",
    "zoom-gym-bmi": "data/training_images/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps/png",
    "static-office-hair": "data/training_images/20221022_3_250_batch01handhair_static_bigOffice_30fps/png",
    "zoom-suburbd-hair": "data/training_images/20221024_10_100_batch01handhair_zoom_suburb_d_30fps/png",
    "static-gym-hair": "data/training_images/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/png",
}

ENV_MASK_PATH = "data/env_masks"
HM_SAMPLES_PATH = "data/heatmap_samples/"

bp = "data/annotations/"  # annotation base path
DATASET_FILES = [
    {  # validation/testing data
        "3dpw-test-cam": join(bp, "3dpw_test_w_cam.npz"),
        "emdb-p1": join(bp, "emdb_p1.npz"),
        "emdb-p1-occl": join(bp, "emdb_p1_occl.npz"),
    },
    {
        # train data
        "zoom-suburbd": join(bp, "20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz"),
        "closeup-suburba": join(bp, "20221011_1_250_batch01hand_closeup_suburb_a_6fps.npz"),
        "closeup-suburbb": join(bp, "20221011_1_250_batch01hand_closeup_suburb_b_6fps.npz"),
        "closeup-suburbc": join(bp, "20221011_1_250_batch01hand_closeup_suburb_c_6fps.npz"),
        "closeup-suburbd": join(bp, "20221011_1_250_batch01hand_closeup_suburb_d_6fps.npz"),
        "closeup-gym": join(bp, "20221012_1_500_batch01hand_closeup_highSchoolGym_6fps.npz"),
        "zoom-gym": join(bp, "20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.npz"),
        "static-gym": join(bp, "20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.npz"),
        "static-office": join(bp, "20221013_3_250_batch01hand_static_bigOffice_6fps.npz"),
        "orbit-office": join(bp, "20221013_3_250_batch01hand_orbit_bigOffice_6fps.npz"),
        "orbit-archviz-15": join(bp, "20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz"),
        "orbit-archviz-19": join(bp, "20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps.npz"),
        "orbit-archviz-12": join(bp, "20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps.npz"),
        "orbit-archviz-10": join(bp, "20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps.npz"),
        "static-hdri": join(bp, "20221010_3_1000_batch01hand_6fps.npz"),
        "static-hdri-zoomed": join(bp, "20221017_3_1000_batch01hand_6fps.npz"),
        "pitchup-stadium": join(bp, "20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps.npz"),
        "static-hdri-bmi": join(bp, "20221019_3_250_highbmihand_6fps.npz"),
        "closeup-suburbb-bmi": join(bp, "20221019_1_250_highbmihand_closeup_suburb_b_6fps.npz"),
        "closeup-suburbc-bmi": join(bp, "20221019_1_250_highbmihand_closeup_suburb_c_6fps.npz"),
        "static-suburbd-bmi": join(bp, "20221019_3-8_1000_highbmihand_static_suburb_d_6fps.npz"),
        "zoom-gym-bmi": join(bp, "20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.npz"),
        "pitchdown-stadium": join(bp, "20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps.npz"),
        "static-office-hair": join(bp, "20221022_3_250_batch01handhair_static_bigOffice_30fps.npz"),
        "zoom-suburbd-hair": join(bp, "20221024_10_100_batch01handhair_zoom_suburb_d_30fps.npz"),
        "static-gym-hair": join(bp, "20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.npz"),
        "agora-bfh": join(bp, "agora-bfh.npz"),
        "agora-body": join(bp, "agora-body.npz"),
        "3dpw-train-smplx": join(bp, "3dpw_train_smplx_w_cam.npz"),
    },
]

# Download the models from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# and update the path
PRETRAINED_CKPT_FOLDER = {
    "hrnet_w48-coco": "data/ckpt/pose_hrnet_w48_256x192.pth",
    # from BEDLAM paper
    "hrnet_w48-bedlam": "data/ckpt/bedlam_cliff_3dpw_ft_backbone.pth",
}
hparams = CN()
# General settings
hparams.LOG_DIR = "logs"
hparams.EXP_NAME = "dummy"
hparams.RUN_TEST = False

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.SCALE_FACTOR = 0.25
hparams.DATASET.CROP_PROB = 0.0
hparams.DATASET.CROP_FACTOR = 0.0
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.EVAL_BATCH_SIZE = 32
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.SHUFFLE_TRAIN = True
hparams.DATASET.VAL_DS = "3dpw-test-cam"
hparams.DATASET.TEST_DS = "3dpw-test-cam"
hparams.DATASET.IMG_RES = 224
hparams.DATASET.TRAIN_DS = "3dpw-train-smplx"
hparams.DATASET.ALB = False
hparams.DATASET.ALB_PROB = 0.3
hparams.DATASET.CONF_VIS_TH = 0.5
hparams.DATASET.FILTER_DATASET = False

# optimizer config
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.LR = 5e-5
hparams.OPTIMIZER.WD = 0.0

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.PRETRAINED_CKPT = None
hparams.TRAINING.RESUME_CKPT = None
hparams.TRAINING.MAX_EPOCHS = 30
hparams.TRAINING.PROG_BAR_FREQ = 100
hparams.TRAINING.FREEZE_BACKBONE = False
hparams.TRAINING.HM_SAMPLE_MAX_TH = 0.70
hparams.TRAINING.HM_SAMPLE_MIN_TH = 0.15
hparams.TRAINING.SMPL_PARAM_NOISE_RATIO = 0.005
hparams.TRAINING.NUM_TRAIN_SAMPLES = 2
hparams.TRAINING.NUM_TEST_SAMPLES = 100
hparams.TRAINING.NUM_VIZ = 2  # number of examples per val/test dataset to visualize

hparams.MODEL = CN()
hparams.MODEL.BACKBONE = "hrnet_w48-conv"

hparams.FREIA_NF = CN()
hparams.FREIA_NF.CLAMP = 3.0
hparams.FREIA_NF.N_BLOCKS = 8
hparams.FREIA_NF.FC_SIZE = 256
hparams.FREIA_NF.DROPOUT = 0.0

hparams.LOSS_WEIGHTS = CN()
hparams.LOSS_WEIGHTS.KEYPOINTS_2D_MODE = 0.01
hparams.LOSS_WEIGHTS.ORTHOGONAL = 0.1
hparams.LOSS_WEIGHTS.BETAS_MODE = 0.0005
hparams.LOSS_WEIGHTS.NLL = 0.1
hparams.LOSS_WEIGHTS.LOCATION_MASK = 0.1
hparams.LOSS_WEIGHTS.KEYPOINTS_2D_HM_SAMPLE = 0.05


def get_hparams_defaults():
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()
