import torch
from torch import nn
from torch.nn import functional as F


class Net_NoSharedWeights(nn.Module):
    def __init__(self, proba, kconv1, kconv2, kconv3, kernel1, kernel2, kernel3):

        super(Net_NoSharedWeights, self).__init__()

        #Convolutional layers
        self.conv1 = nn.Conv2d(2, kconv1, kernel_size=kernel1)
        self.conv1_bn = nn.BatchNorm2d(kconv1)

        self.conv2 = nn.Conv2d(kconv1, kconv2, kernel_size=kernel2)
        self.conv2_bn = nn.BatchNorm2d(kconv2)

        self.conv3 = nn.Conv2d(kconv2, kconv3, kernel_size=kernel3)
        self.conv3_bn = nn.BatchNorm2d(kconv3)

        self.drop1 = nn.Dropout(p=proba)

        #Calculations for the linear layer
        after1 = (14 - kernel1 + 1) / 2
        after2 = after1 - kernel2 + 1
        after3 = after2 - kernel3 + 1

        #Prediction of each numbers
        self.fc1 = nn.Linear(int(after3 * after3 * kconv3), 180)
        self.fc2 = nn.Linear(180, 20)
        #Prediction of the comparison of the numbers
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x, kernel1, kernel2, kernel3, kconv3):
        #Convolution layers
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), kernel_size=2))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))

        x = self.drop1(x)

        #Calculations for the linear layer
        after1 = (14 - kernel1 + 1) / 2
        after2 = after1 - kernel2 + 1
        after3 = after2 - kernel3 + 1

        #Prediction of each number through the two first fully connected layers
        x = F.relu(self.fc1(x.view(-1, int(after3 * after3 * kconv3))))
        x = self.fc2(x)
        numbers = x

        #Prediction of the comparison of the two numbers in the two last fully connected layers
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        comparison = x

        #output to be returns contains the prediction of the comparison and the numbers predicted 
        output_list = [comparison, numbers]
        output = torch.cat(output_list, dim=1)

        return output
