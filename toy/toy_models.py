import torch
from torch.nn import Parameter
import torch.nn.functional as F


class ScoreNetwork(torch.nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim=1, activate_func=F.relu):

        super(ScoreNetwork, self).__init__()
        self.num_layers = num_layers
        self.activate_func = activate_func

        self.linears = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.linears.append( torch.nn.Linear(input_dim, hidden_dim) )
            else:
                self.linears.append( torch.nn.Linear(hidden_dim, hidden_dim) )
        
        for _ in range(self.num_layers):
            if _==self.num_layers-1:
                self.linears.append( torch.nn.Linear(hidden_dim, output_dim) )
            else:
                self.linears.append( torch.nn.Linear(hidden_dim, hidden_dim) )

    def forward(self, x):

        h = x
        for _ in range(len(self.linears)):
            res = h
            h = self.linears[_](h)
            if _ < len(self.linears)-1:
                h = self.activate_func(h)
                if _>0: 
                    h = h + res
        return h
        