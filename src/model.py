from torch import nn
import torch

class ConvBlock(nn.Module): # this is a helper class for the convolutional layers that does the work we want to on each layer 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): #copilot wrote this
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class LinearBlock(nn.Module): # this is the helper class for the linear layers
    def __init__(self, input, output):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input, output) # could add dropout
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class MatchPredictor(nn.Module): # this is the match predictor model
    def __init__(self): # input size is a 28 long vector 
        super(MatchPredictor, self).__init__() # imports the features of nn.Module
        self.dropout = nn.Dropout(0.2)
        self.conv1 = ConvBlock(1, 16, kernel_size=3, stride=1, padding=0) # first conv layer; Conv --> 26 
        self.conv2 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=0) # by adding a second conv layer we extract more interesting details; Conv --> 24      

        self.batchnorm = nn.BatchNorm1d(32)

        self.fc1 = LinearBlock(32 * 24 * 1, 64)
        self.fc2 = LinearBlock(64, 128)  # this is now 1D, so no channels
        
        self.output = nn.Linear(128, 2) # the output layer

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        (B, C, W) = x.shape
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x) 
        output = self.output(x)
        return output


    
