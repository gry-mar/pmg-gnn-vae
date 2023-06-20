import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pyro
import pyro.distributions as dist
from math import log
from models.helper import log_normal_diag, log_bernoulli

    

class Encoder(nn.Module):
    def __init__(self, 
                 n_input_features: int,
                 n_hidden_neurons: int,
                 n_latent_features: int):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(n_input_features, n_hidden_neurons)
        self.fc21 = nn.Linear(n_hidden_neurons, n_latent_features)
        self.fc22 = nn.Linear(n_hidden_neurons, n_latent_features)
        

    def encode(self, x):
        if isinstance(x, torch.DoubleTensor):
            h1 = nn.functional.relu(self.fc1(x.type(torch.FloatTensor)))
        else:
            h1 = nn.functional.relu(self.fc1(x))
        mu_e = self.fc21(h1)
        log_var_e = torch.clamp(self.fc22(h1), max=log(5))
        return mu_e, log_var_e 

    def sample(self, x=None, mu_e=None, log_var_e=None):
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log cant be none')
            # z = self.reparameterization(mu_e, log_var_e)
            z = pyro.sample("latent", dist.Normal(mu_e, log_var_e).to_event(1))
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-scale and z can`t be None!')
        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)
    

class Decoder(nn.Module):
    def __init__(self, 
                 n_latent_features: int,
                 n_hidden_neurons: int,
                 n_output_features: int, 
                 ):
        super(Decoder, self).__init__()

        self.latent_to_hidden = nn.Linear(n_latent_features, n_hidden_neurons)
        self.hidden_to_output = nn.Linear(n_hidden_neurons, n_output_features)

    def decode(self, z):
        h1 = nn.functional.relu(self.latent_to_hidden(z))
        r = torch.sigmoid(self.hidden_to_output(h1))
        mu_d = torch.sigmoid(r)
        return [mu_d]

    def sample(self, z):
        outs = self.decode(z)
        mu_d = outs[0]
        x_new = torch.bernoulli(mu_d)
        return x_new

    def log_prob(self, x, z):
        outs = self.decode(z)
        mu_d = outs[0]
        log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)
        return log_p

    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(x)



class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, L=16):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.qz = None
        self.L = L
    
    def forward(self, x, reduction='avg'):
        x = x.view(x.shape[0], -1)
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=torch.exp(log_var_e))
        
        # ELBO
        RE = self.decoder.log_prob(x, z)
        if self.prior == 'normal':
            self.qz = torch.distributions.Normal(mu_e, torch.exp(log_var_e))
            KL = torch.distributions.kl_divergence(self.qz, torch.distributions.Normal(0, 1)).sum(-1).mean()
        else:
            KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=torch.exp(log_var_e), z=z)).sum(-1)

        
        error = 0
        if np.isnan(RE.detach().numpy()).any():
            print('RE {}'.format(RE))
            error = 1
        if np.isnan(KL.detach().numpy()).any():
            print('KL {}'.format(KL))
            error = 1

        if error == 1:
            raise ValueError()

        if reduction == 'sum':
            return -(RE + KL).sum()
        else:
            return -(RE + KL).mean()

    def sample(self, batch_size=64, is_decoder=True):
        if self.prior == 'normal':
            z = self.qz.sample()
        else:
            z = self.prior.sample(batch_size=batch_size)
        if not is_decoder:
            return z
        else:
            return self.decoder.sample(z)