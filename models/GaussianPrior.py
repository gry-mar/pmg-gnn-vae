import torch
from models.helper import log_standard_normal
import torch.nn as nn

class StandardPrior(nn.Module):
    def __init__(self, L=2):
        super(StandardPrior, self).__init__()

        self.L = L 

        # params weights
        self.means = torch.zeros(1, L)
        self.logvars = torch.zeros(1, L)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        if not isinstance(batch_size, int):
            batch_size = 64
        return torch.randn(batch_size, self.L)
    
    def log_prob(self, z):
        return log_standard_normal(z)