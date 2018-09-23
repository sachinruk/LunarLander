import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        h = 20
        self.model = nn.Sequential(nn.Linear(state_size, h),
                      nn.ReLU(),
                      nn.Linear(h, h),
                      nn.ReLU(),
                      nn.Linear(h, action_size)
                      )
        self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(model.parameters())

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model.forward(state)
