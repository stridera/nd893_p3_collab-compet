

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from .ddpg import DDPGAgent
from .ReplayBuffer import ReplayBuffer


class MADDPG:
    def __init__(self, state_size: int, action_size: int, num_agents: int = 1, seed: int = 0, discount_factor=0.95,
                 tau=0.02, writer: SummaryWriter = None) -> None:

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents, seed, device=self.device)
                             for _ in range(num_agents)]

        self.num_agents = num_agents
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.batch_size = 256
        self.gamma = 0.99
        self.learn_every = 2*num_agents
        buffer_size = int(1e6)
        self.noise = 0
        self.noise_decay = 0.99

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, self.device, seed)

        self.t_step = 0
        self.writer = writer

    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset_noise()

    def step(self, obs, actions, rewards, next_obs, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.push(obs, actions, rewards, next_obs, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and self.t_step % self.learn_every == 0:
            for idx in range(len(self.maddpg_agent)):
                experiences = self.memory.sample(self.batch_size)
                self.update(experiences, idx)
            self.update_targets()

        self.t_step += 1

    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, self.noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        self.noise *= self.noise_decay
        return actions

    def target_act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        stacked = torch.stack(obs_all_agents, axis=1)
        actions = [agent.target_act(obs, self.noise) for agent, obs in zip(self.maddpg_agent, stacked)]
        self.noise *= self.noise_decay
        return actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions)
        target_actions = torch.reshape(target_actions, (-1, agent.action_size * self.num_agents))

        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full, target_actions)

        reward = torch.stack(reward, dim=1)
        done = torch.stack(done, dim=1)
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))

        action = torch.cat(action)
        action = torch.reshape(action, (-1, agent.action_size * self.num_agents))
        q = agent.critic(obs_full, action)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        obs = torch.stack(obs, dim=1)
        q_input = [self.maddpg_agent[i].actor(ob) if i == agent_number
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs)]

        q_input = torch.cat(q_input, dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        if self.writer is not None:
            actor_loss = actor_loss.cpu().detach().item()
            critic_loss = critic_loss.cpu().detach().item()
            self.writer.add_scalar('loss/actor', actor_loss, self.t_step)
            self.writer.add_scalar('loss/critic', critic_loss, self.t_step)

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            self.soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            self.soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)

    def save_model(self, path):
        for idx, agent in enumerate(self.maddpg_agent):
            agent.save_model(path + f"/agent{idx}/")

    def load_model(self, path):
        for idx, agent in enumerate(self.maddpg_agent):
            agent.load_model(path + f"/agent{idx}/")

    @ staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
