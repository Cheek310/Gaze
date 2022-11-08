import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, input_channels):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.maxpool(self.relu(self.conv2(x1)))
        x3 = self.relu(self.conv3(x2))
        x4 = self.maxpool(self.relu(self.conv4(x3)))
        x5 = self.relu(self.conv5(x4))
        x6 = self.maxpool(self.relu(self.conv6(x5)))
        out = x6.view(x.size(0), 256*1*4)
        return out


class ENet(nn.Module):
    def __init__(self, input_channels):
        super(ENet, self).__init__()
        self.stream1 = nn.Sequential(
            BaseCNN(input_channels),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 1 * 4, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(inplace=True),
        )
        self.stream2 = nn.Sequential(
            BaseCNN(input_channels),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 1 * 4, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(in_features=1000, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_left, x_right):
        x1 = self.stream1(x_left)
        x2 = self.stream2(x_right)
        x = torch.cat((x1, x2), dim=1)
        out = self.softmax(self.linear(x))
        return out


class ARNet(nn.Module):
    def __init__(self, input_channels):
        super(ARNet, self).__init__()
        self.stream1 = nn.Sequential(
            BaseCNN(input_channels),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 1 * 4, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(inplace=True),
        )
        self.stream2 = nn.Sequential(
            BaseCNN(input_channels),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 1 * 4, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(inplace=True),
        )
        self.stream3 = nn.Sequential(
            BaseCNN(input_channels),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 1 * 4, out_features=500),
            nn.ReLU(inplace=True),
        )
        self.stream4 = nn.Sequential(
            BaseCNN(input_channels),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 1 * 4, out_features=500),
            nn.ReLU(inplace=True),
        )
        self.linear1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Linear(in_features=1506, out_features=3)

    def forward(self, x_left, x_right, headpose):
        x1 = self.stream1(x_left)
        x2 = self.stream2(x_right)
        x3 = self.stream3(x_left)
        x4 = self.stream4(x_right)
        x34 = torch.cat((x3, x4), dim=1)
        x34 = self.linear1(x34)
        x = torch.cat((x1, x2, x34, headpose), dim=1)
        out = self.linear2(x)

        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    model = ARNet(1)
    model.to(device)
    model.eval()
    x = torch.randn(1, 1, 36, 60)
    x = x.to(device)
    h = torch.randn(1, 6)
    h = h.to(device)
    out = model(x, x, h)
    print(out.shape)
