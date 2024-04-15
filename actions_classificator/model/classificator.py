import torch
import torch.nn as nn


class ActionsClassificator(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fc_01 = nn.Linear(in_features=32, out_features=64)
        self.fc_02 = nn.Linear(in_features=64, out_features=128)
        self.fc_03 = nn.Linear(in_features=128, out_features=64)
        self.fc_04 = nn.Linear(in_features=64, out_features=32)
        self.fc_05 = nn.Linear(in_features=32, out_features=16)
        self.fc_06 = nn.Linear(in_features=16, out_features=num_classes)

        self.activation = nn.ReLU()

        self.out_layer = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc_01(x)
        x = self.activation(x)

        x = self.fc_02(x)
        x = self.activation(x)

        x = self.fc_03(x)
        x = self.activation(x)

        x = self.fc_04(x)
        x = self.activation(x)

        x = self.fc_05(x)
        x = self.activation(x)

        x = self.fc_06(x)

        x = self.out_layer(x)

        return x
