#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Train the given environment using PPO. """

import argparse
from collections import deque
from typing import Optional
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from ddpg import Agent
# from maddpg import Agent


def main(env: str, episodes: int = 10000, seed: Optional[int] = None, max_steps: int = 10000) -> None:
    print(f"Using environment: {env}")
    env = UnityEnvironment(file_name=env)
    seed = seed if seed is not None else np.random.randint(0, 10000)

    writer = SummaryWriter()

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    print(f"Seed: {seed}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Number of agents: {num_agents}")

    agent = Agent(state_size, action_size, num_agents=num_agents, seed=seed, writer=writer)

    ep_avg_scores = deque(maxlen=100)
    progress = tqdm(range(1, episodes + 1), desc="Training", ncols=120)
    solved = False
    for i_episode in progress:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()

        for step in range(max_steps):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        ep_score = np.max(scores)
        ep_avg_scores.append(ep_score)
        writer.add_scalar("score/score", ep_score, i_episode)
        writer.add_scalar("score/last_100_scores", np.mean(ep_avg_scores), i_episode)

        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode} Score: {ep_score:.4f} Last 100 avg score: {np.mean(ep_avg_scores):.4f}",
                  ' ' * 120)
            agent.save_model(f"checkpoints/checkpoint_{i_episode}/")

        progress.set_postfix(ep_score=ep_score, last_100_avg=np.mean(ep_avg_scores))

        if not solved and np.mean(ep_avg_scores) > 0.5:
            print(
                f"Solved in {i_episode} episodes with a last 100 episode avg score of {np.mean(ep_avg_scores):.2f}!")
            agent.save_model("checkpoints/solved/")
            solved = True

    # Save final model
    print(f"Final score of last 100 episodes: {np.mean(ep_avg_scores)}")
    agent.save_model("checkpoints/final/")

    env.close()
    writer.close()


if __name__ == "__main__":
    default_path = 'unity_env/Tennis_Linux/Tennis.x86_64'
    parser = argparse.ArgumentParser(description='DQN Agent')
    parser.add_argument('--env', type=str, default=default_path,
                        help=f'Path to the environment.  Default: {default_path}')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to run.  Default: 600')
    parser.add_argument('--seed', type=int, required=False, help='Random seed. Default: None (random)')

    args = parser.parse_args()
    main(args.env, args.episodes, args.seed)
