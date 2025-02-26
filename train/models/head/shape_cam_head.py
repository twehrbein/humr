import torch
import torch.nn as nn
import numpy as np
from train.core.constants import NUM_BETAS
from train.core.config import SMPL_MEAN_PARAMS


class HMRShapeCamHead(nn.Module):
    def __init__(
        self,
        num_input_features,
    ):
        super(HMRShapeCamHead, self).__init__()

        self.num_input_features = num_input_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_input_features + NUM_BETAS + 3, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        self.decshape = nn.Linear(1024, NUM_BETAS)
        self.deccam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_shape_ = torch.from_numpy(mean_params["shape"][:].astype("float32")).unsqueeze(0)
        init_shape = torch.cat((init_shape_, torch.zeros((1, 1))), -1)
        init_cam = torch.from_numpy(mean_params["cam"]).unsqueeze(0)

        self.register_buffer("init_shape", init_shape)
        self.register_buffer("init_cam", init_cam)

    def forward(self, features, j2d_emb, bbox_info, n_iter=3):

        batch_size = j2d_emb.shape[0]
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        if features.ndim > 2:
            xf = self.avgpool(features)
            xf = xf.view(xf.size(0), -1)
        else:
            xf = features

        pred_shape = init_shape
        pred_cam = init_cam
        for _ in range(n_iter):
            xc = torch.cat([xf, j2d_emb, bbox_info, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        output = {
            "pred_cam": pred_cam,
            "pred_shape": pred_shape,
            "body_feat": xf,
        }

        return output
