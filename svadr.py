import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Enhance the SVADR classification model
class SVADR(nn.Module):
    def __init__(self, input_size, n):
        super(SVADR, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (input_size // 4), n)
        self.fc2 = nn.Linear(n, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.T
        # return torch.round(x).T # TODO implement thresholding during evaluation