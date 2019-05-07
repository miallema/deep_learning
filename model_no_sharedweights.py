import torch
from torch import nn
from torch.nn import functional as F


class Net_NoSharedWeights(nn.Module):
    def __init__(self, random_init, proba, kconv1, kconv2, kconv3, kernel1, kernel2, kernel3):

        super(Net_NoSharedWeights, self).__init__()

        self.conv1 = nn.Conv2d(2, kconv1, kernel_size=kernel1)
        if random_init:
            # default is generally kaiming_uniform_ or Xavier
            torch.nn.init.kaiming_uniform_(self.conv1.weight)
            self.conv1.bias.data.fill_(0.01)
        self.conv1_bn = nn.BatchNorm2d(kconv1)

        self.conv2 = nn.Conv2d(kconv1, kconv2, kernel_size=kernel2)
        if random_init:
            # default is generally kaiming_uniform_ or Xavier
            torch.nn.init.kaiming_uniform_(self.conv2.weight)
            self.conv2.bias.data.fill_(0.01)
        self.conv2_bn = nn.BatchNorm2d(kconv2)

        self.conv3 = nn.Conv2d(kconv2, kconv3, kernel_size=kernel3)
        if random_init:
            # default is generally kaiming_uniform_ or Xavier
            torch.nn.init.kaiming_uniform_(self.conv3.weight)
            self.conv3.bias.data.fill_(0.01)
        self.conv3_bn = nn.BatchNorm2d(kconv3)

        self.drop1 = nn.Dropout(p=proba)

        after1 = (14 - kernel1 + 1) / 2
        after2 = after1 - kernel2 + 1
        after3 = after2 - kernel3 + 1

        self.fc1 = nn.Linear(int(after3 * after3 * kconv3), 180)
        if random_init:
            # default is generally kaiming_uniform_ or Xavier
            torch.nn.init.kaiming_uniform_(self.fc1.weight)
            self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(180, 20)
        if random_init:
            # default is generally kaiming_uniform_ or Xavier
            torch.nn.init.kaiming_uniform_(self.fc2.weight)
            self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(20, 100)
        if random_init:
            # default is generally kaiming_uniform_ or Xavier
            torch.nn.init.kaiming_uniform_(self.fc3.weight)
            self.fc3.bias.data.fill_(0.01)

        self.fc4 = nn.Linear(100, 2)
        if random_init:
            # default is generally kaiming_uniform_ or Xavier
            torch.nn.init.kaiming_uniform_(self.fc4.weight)
            self.fc4.bias.data.fill_(0.01)

    def forward(self, x, kernel1, kernel2, kernel3, kconv3):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), kernel_size=2))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))

        x = self.drop1(x)

        after1 = (14 - kernel1 + 1) / 2
        after2 = after1 - kernel2 + 1
        after3 = after2 - kernel3 + 1

        x = F.relu(self.fc1(x.view(-1, int(after3 * after3 * kconv3))))
        x = self.fc2(x)
        numbers = x

        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        comparison = x

        output_list = [comparison, numbers]
        output = torch.cat(output_list, dim=1)

        return output
