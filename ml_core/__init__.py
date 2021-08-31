import cv2
from numpy import np
from scipy import ndimage
from torch import nn


def preprocess(x, resize=128):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (resize * 2, resize * 2))
    x = cv2.GaussianBlur(x, (5, 5), 1, 1)
    x = cv2.resize(x, (resize, resize))
    x = x - (cv2.GaussianBlur(x, (7, 7), 1, 1) * 0.8)
    x = (x - x.mean()).astype(np.float64)
    x = (x / x.max()).astype(np.float64)
    x = x.reshape(resize, resize, 1)
    return x


class PneumoniaModel(nn.Module):
    def __init__(self, input_shape, dropout=0.1):
        super(PneumoniaModel, self).__init__()
        n_filters = 2
        self.conv1 = nn.Conv2d(1, n_filters, 3, 1)
        self.pool1 = nn.MaxPool2d((3, 3), stride=1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(n_filters, n_filters, (3, 3), 1)
        self.pool2 = nn.MaxPool2d((3, 3), stride=1)
        self.dropout2 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_filters * (input_shape[0] - 8) * (input_shape[1] - 8), 32 * 32)
        self.activation1 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(32 * 32, 32 * 32)
        self.activation2 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(32 * 32, 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, out=0):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        if out == -2:
            return x
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        if out == -1:
            return x
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout4(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
