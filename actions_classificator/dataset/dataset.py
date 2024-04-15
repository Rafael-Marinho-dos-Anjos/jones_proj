import numpy as np
from PIL import Image
from os import listdir
from torch import float32, Tensor, zeros
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, Normalize, InterpolationMode

from pose_estimation.pose_est import estimate


class MyDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()

        self.path = path
        self.im_names = [name for name in listdir(self.path) if name[:4] in ["comp", "sent", "empe"]]

    def __getitem__(self, index):
        name = self.im_names[index]
        image = Image.open("{}/{}".format(self.path, name))
        
        pose = estimate(image)['body']

        x = np.array([point[0] for point in pose])
        x = x - x.mean()
        x = x / np.std(x)
        y = np.array([point[1] for point in pose])
        y = y - y.mean()
        y = y / np.std(y)

        pose = np.concatenate((x, y), axis=0)
        pose = Tensor(pose)

        label = zeros((3, ), dtype=float32)

        if name[:4] == "comp":
            label[0] = 1
        elif name[:4] == "sent":
            label[1] = 1
        elif name[:4] == "empe":
            label[2] = 1
    
        return pose, label
    
    def __len__(self):
        return len(self.im_names)


if __name__ == "__main__":
    PATH = r"data\sample_images"
    ds = MyDataset(PATH)

    print(ds[0])