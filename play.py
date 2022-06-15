#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to play the game with the specified model. """

import argparse
import numpy as np
from unityagents import UnityEnvironment

from ddpg import Agent


def main(env: str, model_path: str, min_steps: int = 300) -> None:
    env = UnityEnvironment(file_name=env)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)

    agent = Agent(state_size, action_size)
    agent.load_model(model_path)
    print(f"Loaded model from {model_path}")

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    step = 0
    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        scores += rewards
        step += 1
        if np.any(dones):
            if step > min_steps:
                break
            env.reset(train_mode=False)[brain_name]

    print(f"Game Over.  Final score: {np.sum(scores):2f}  Avg Score: {np.mean(scores):2f}")

    env.close()


if __name__ == "__main__":
    default_path = 'unity_env/Tennis_Linux/Tennis.x86_64'

    parser = argparse.ArgumentParser(description='DQN Agent')
    parser.add_argument('--env', type=str, default=default_path,
                        help='Path to the environment.  Default: {default_path}')
    parser.add_argument('--model_path', type=str, default='checkpoints/final/',
                        help='Path to the model.  Default: checkpoints/final/')
    parser.add_argument('--min_steps', type=int, default=300,
                        help='Min number of steps to compete before quitting  Default: 300')

    args = parser.parse_args()
    main(args.env, args.model_path, args.min_steps)
