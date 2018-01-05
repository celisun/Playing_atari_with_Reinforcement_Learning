import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """  Convolutoin net with three X {conv -> bn -> relu} followed by a final fully connected to generate the output.
        Args:       
            dim1, dim2: the height and width of the frame/state input
            output_size: the number of actions in action space        
        Return: 
            the estimated Q values of for each action
    """

    def __init__(self, dim1, dim2, output_size, kernel_size=5, stride=2):      #RGB input size dim1 dim2
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        self.bn3 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(dim1*dim2*64, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        return self.linear(x.view(x.size(0), -1))

    
    
    
    
