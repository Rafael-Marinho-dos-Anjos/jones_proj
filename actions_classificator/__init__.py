from torch import load

from actions_classificator.model.classificator import ActionsClassificator
from actions_classificator.dataset.dataset import MyDataset


ds = MyDataset(r"data\sample_images")

def get_model(pretrained = False):
    model = ActionsClassificator(num_classes=3)

    if pretrained:
        model.load_state_dict(load(r"actions_classificator\weights\weights.pth"))
