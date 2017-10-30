import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    """
        Simple multiple layer net, with 5 hidden layers
        each equipped with rectified non-linearity.
        
    """
    
    def __init__ (self, input_dim, output_dim, hidden_dim, hidden_n=5):
        super(net, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.hidden_n = hidden_n
        self.input_dim = input_dim
                   
        
    def forward(self, x):
        assert x.data.size()[1]* x.data.size()[2]* x.data.size()[3] == self.input_dim
        x = x.view(-1, self.input_dim)
        
        x = F.relu(self.input_layer(x))
        for i in range(self.hidden_n):
            x = F.relu(self.hidden_layer(x))
        x = F.relu(self.output_layer(x))
        
        return x
        