import cv2
import torch
import numpy as np

from actions_classificator import get_model
from face_detection.face_det import detect
from pose_estimation.pose_est import estimate


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model(pretrained=True).to(device)

labels = [
     "computador",
     "sentado",
     "em pé"
     ]

def norm_pose(pose):
    x = np.array([point[0] for point in pose])
    x = x - x.mean()
    x = x / np.std(x)
    y = np.array([point[1] for point in pose])
    y = y - y.mean()
    y = y / np.std(y)

    pose = np.concatenate((x, y), axis=0)
    pose = torch.Tensor(pose)

    return pose


def look_person_activitie(image, thresh = 25):
    pose = estimate(image)
    face = detect(image)

    for instace in face[1]:
        head = pose["head"]
        dist = ((instace[0] - head[0])**2 + (instace[1] - head[1])**2) ** -1

        if dist <= thresh:
            points = norm_pose(pose["body"]).to(device)
            pred = model(points)
            action = torch.argmax(pred, dim=0).item()
            face[0] = cv2.putText(
                face[0], labels[action], head,
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)

    return face[0]
