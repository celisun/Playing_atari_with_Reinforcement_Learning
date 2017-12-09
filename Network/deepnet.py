import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
        Simple multiple layer net, with 5 hidden layers
        each equipped with a rectified non-linearity.
        
    """
    
    def __init__ (self, input_dim, output_dim, hidden_dim):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_5 = nn.Linear(hidden_dim, hidden_dim)     
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        
        self.bn = nn.BatchNorm2d(hidden_dim)

                   
        
    def forward(self, x):
        #print str(x.data.size())
        #x = F.relu(self.input_layer(x))
        x = self.tanh(self.input_layer(x))
        #x = x = self.bn(x)
     
        x = self.tanh(self.hidden_layer_1(x))
        #x = F.dropout(x)
        #x = F.relu(self.hidden_layer_1(x))
        #x = self.bn(x)
        x = self.tanh(self.hidden_layer_2(x))
        #x = F.relu(self.hidden_layer_2(x))
        #x = self.bn(x)
        x = self.tanh(self.hidden_layer_3(x))
        #x = F.relu(self.hidden_layer_3(x))
        #x = self.bn(x)
        x = self.tanh(self.hidden_layer_4(x))
        #x = F.relu(self.hidden_layer_4(x))
        #x = self.bn(x)
        x = self.tanh(self.hidden_layer_5(x))
        #x = self.bn(x)
        #x = F.relu(self.hidden_layer_5(x))
        
        x = self.output_layer(x)
        #x = self.Tanh(x)
 
        
        return x
        
