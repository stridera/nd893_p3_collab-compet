
import os
import torch
from torch.optim import Adam

from .OUNoise import OUNoise
# from .networkforall import Network
from .model import Actor, Critic


class DDPGAgent:
    def __init__(self, state_size: int, action_size: int, num_agents: int = 1, seed: int = 0,
                 lr_actor=1.0e-2, lr_critic=1.0e-2, device=torch.device('cpu')) -> None:
        """ Initialize parameters and build model.

        Params
        ======
            in_actor (int): Dimension of each state
            hidden_in_actor (int): Number of nodes in first hidden layer
            hidden_out_actor (int): Number of nodes in second hidden layer
            out_actor (int): Dimension of each action
            in_critic (int): Dimension of each state
            hidden_in_critic (int): Number of nodes in first hidden layer
            hidden_out_critic (int): Number of nodes in second hidden layer
            lr_actor (float): Learning rate of the actor network
            lr_critic (float): Learning rate of the critic network
        """

        self._action_size = action_size
        self.device = device
        self.actor = Actor(state_size, action_size, seed).to(self.device)
        self.target_actor = Actor(state_size, action_size, seed).to(self.device)

        self.critic = Critic(state_size*num_agents, action_size*num_agents, seed).to(self.device)
        self.target_critic = Critic(state_size*num_agents, action_size*num_agents, seed).to(self.device)

        print(self.actor, self.critic)

        self.noise = OUNoise(action_size, scale=1.0, device=self.device)

        # initialize targets same as original networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, obs, noise=0.0):
        obs = torch.from_numpy(obs).float().to(self.device)
        action = self.actor(obs) + noise*self.noise.noise()
        return action.cpu().data.numpy()

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action  # .cpu().data.numpy()

    def reset_noise(self):
        self.noise.reset()

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Params
        ======
            path (str): path to save the model
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic.state_dict(), path + 'critic.pth')

    def load_model(self, path: str) -> None:
        """Load the model from a file.

        Params
        ======
            path (str): path to load the model
        """
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'critic.pth'))

    @property
    def action_size(self):
        return self._action_size

    @staticmethod
    def hard_update(target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
