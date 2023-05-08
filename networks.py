from torch import nn
import efficientunet
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, 5)        # out = 60
        self.conv2 = nn.Conv2d(10, 10, 3)       # in = 30, out = 14

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(10 * 14 * 14, 128)
        self.o_n = nn.Linear(128, 1)


        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()

    def forward(self, inpt):
        out = self.activation(self.conv1(inpt))
        out = self.pool(out)
        
        out = self.activation(self.conv2(out))
        out = self.pool(out)

        out = self.flatten(out)

        out = self.activation(self.fc1(out))
        out = self.o_n(out)

        return out

class EfficientUNet_B0(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = efficientunet.get_efficientunet_b0(out_channels=3, concat_input=True, pretrained=True)

    def forward(self, inpt):
        out = self.main(inpt)
        out = F.sigmoid(out)
        return out