import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
                                        nn.BatchNorm2d(32),
                                        nn.MaxPool2d(kernel_size=(2, 2)),
                                        nn.ReLU())

        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=(2, 2)),
                                        nn.ReLU())

        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU())

        
        self.linear_layer = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=1152, out_features=100),
                                        nn.ReLU(),
                                        nn.Linear(in_features=100, out_features=10),
                                        nn.Softmax())

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.linear_layer(x)

        return x