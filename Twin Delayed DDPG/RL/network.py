import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def hidden_unit(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    Initialize parameters and build model.
    
    :param state_dim (int): Dimension of each state
    :param action_dim (int): Dimension of each action
    :param max_action (float): highest action to take
    :param seed (int): Random seed
    :param h1_units (int): Number of nodes in first hidden layer
    :param h2_units (int): Number of nodes in second hidden layer
        
    
    :return x: action output of network with tanh activation
    """
    
    def __init__(self, state_dim, action_dim, max_action, seed, h1_units=400, h2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, h1_units)
        self.l2 = nn.Linear(h1_units, h2_units)
        self.l3 = nn.Linear(h2_units, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) 
        return x


class Critic(nn.Module):
    """
    Initialize parameters and build model.

    :param state_dim (int): Dimension of each state
    :param action_dim (int): Dimension of each action
    :param max_action (float): highest action to take
    :param seed (int): Random seed
    :param h1_units (int): Number of nodes in first hidden layer
    :param h2_units (int): Number of nodes in second hidden layer
    
    :return x: value output of network 
    """
    
    def __init__(self, state_dim, action_dim, seed, h1_units=400, h2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, h1_units)
        self.l2 = nn.Linear(h1_units, h2_units)
        self.l3 = nn.Linear(h2_units, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, h1_units)
        self.l5 = nn.Linear(h1_units, h2_units)
        self.l6 = nn.Linear(h2_units, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], dim=1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1