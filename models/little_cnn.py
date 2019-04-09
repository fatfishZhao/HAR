import torch
import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LittleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(LittleCNN, self).__init__()
        self.num_classes = 5
        # self.conv0 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5,5), stride=1, padding=0)
        self.conv0 = BasicConv2d(in_planes=1, out_planes=5, kernel_size=3, stride=2, padding=0)
        self.subsample0 = nn.MaxPool2d(kernel_size=(4,4), stride=4, padding=0)
        # self.conv1 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(5,5), stride=1, padding=0)
        self.conv1 = BasicConv2d(in_planes=5, out_planes=10, kernel_size=(5,5), stride=1, padding=0)
        self.subsample1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(120, self.num_classes)

    def logits(self, features):

        x = features.view(features.size(0), -1)
        x = self.fc(x)
        return x


    def forward(self, x):
        x = self.conv0(x)
        x = self.subsample0(x)
        x = self.conv1(x)
        x = self.subsample1(x)
        x = self.logits(x)
        print(x.shape)
        return x

if __name__ == '__main__':
    model = LittleCNN(num_classes=11)
    input = torch.autograd.Variable(torch.randn(2, 1, 36, 68))
    y = model(input)
