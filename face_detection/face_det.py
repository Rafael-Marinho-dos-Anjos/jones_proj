from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy
from types import MethodType
from os import listdir

from face_detection import torch, device


def encode(img):
    res = resnet(torch.Tensor(img))
    return res


def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces


def detect(img, thres=0.8):
    boxes, cropped_images = mtcnn.detect_box(img)

    if cropped_images is not None:
        instances = {}
        i = 0
        for box, cropped in zip(boxes, cropped_images):
            x, y, x2, y2 = [int(x) for x in box]
            img_embedding = encode(cropped.unsqueeze(0))
            detect_dict = {}
            for k, v in known_faces.items():
                detect_dict[k] = (v - img_embedding).norm().item()
            min_key = min(detect_dict, key=detect_dict.get)

            if detect_dict[min_key] >= thres:
                min_key = 'Undetected'
            
            instances[i] = [((x + x2)/2, (y + y2)/2), min_key] # dict with all faces center and label
            
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img, min_key, (x + 5, y + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
            
            i += 1
                
        return img, instances


mtcnn = MTCNN(select_largest=False, device=device, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

faces_images = "data/known_faces/"
known_faces = {}

for file in listdir(faces_images):
    person_face, extension = file.split(".")
    img = cv2.imread(f'{faces_images}/{person_face}.jpg')
    cropped = mtcnn(img)
    if cropped is not None:
        known_faces[person_face] = encode(cropped)[0, :]

mtcnn.detect_box = MethodType(detect_box, mtcnn)

