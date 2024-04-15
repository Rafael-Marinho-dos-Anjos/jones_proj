import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as T
import cv2
import re
import os
import copy
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import human_pose_estimation.lib.models as models
from human_pose_estimation.lib.core.config import config
from human_pose_estimation.lib.core.config import update_config
from human_pose_estimation.lib.core.config import update_dir
from human_pose_estimation.lib.core.config import get_model_name


CONFIG_FILE = 'human_pose_estimation/experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml'
MODEL_PATH = 'human_pose_estimation\pose_resnet_50_256x256.pth.tar'

model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(config, is_train=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

def estimate(image):
    transform = T.Compose([
                        T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])
                        
    tr_img = transform(image)

    output = model(tr_img.unsqueeze(0))
    output = output.squeeze(0)

    _, OUT_HEIGHT, OUT_WIDTH = output.shape

    # helper function we will use later
    get_detached = lambda x: copy.deepcopy(x.cpu().detach().numpy())

    POSE_PAIRS = [[9, 8],[8, 7],[7, 6],[6, 2],[2, 1],[1, 0],[6, 3],[3, 4],[4, 5],[7, 12],[12, 11],[11, 10],[7, 13],[13, 14],[14, 15]]

    from operator import itemgetter
    get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])

    JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
    JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]

    layers = get_detached(output)
    key_points = list(get_keypoints(pose_layers=layers))

    key_points = [point[1] for point in key_points]

    head = (key_points[9][0] + key_points[8][0])/2, (key_points[9][1] + key_points[8][1])/2

    return {'head': head, "body": key_points}


if __name__ == "__main__":
    image = Image.open(r"sample_images\will-smith-fam-oscars-red-carpet-2022-billboard-1548.jpg")
    image = image.convert("RGB")
    print(estimate(image))
