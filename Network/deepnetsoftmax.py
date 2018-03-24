import torch.nn as nn
import torch.nn.functional as F

class Netsoftmax(nn.Module):
    """
        Simple multiple layer net, with 5 hidden layers
        each equipped with a rectified non-linearity. """
    
    def __init__ (self, input_dim, output_dim, hidden_dim):
        super(Netsoftmax, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_5 = nn.Linear(hidden_dim, hidden_dim)     
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        #self.softmax= nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
   
    def forward(self, x):
        x = self.tanh(self.input_layer(x))
        
        x = self.tanh(self.hidden_layer_1(x))
        x = self.tanh(self.hidden_layer_2(x))
        x = self.tanh(self.hidden_layer_3(x))
        x = self.tanh(self.hidden_layer_4(x))
        x = self.tanh(self.hidden_layer_5(x))
        
        x = self.output_layer(x)
        x = self.log_softmax(x)
    
        return x
        
