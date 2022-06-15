import numpy as np
import random
from collections import deque, namedtuple
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, device, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        def to_tensor(x):
            return torch.from_numpy(np.array(x)).float().to(self.device)

        states = list(map(to_tensor, [e.state for e in experiences if e is not None]))
        all_states = list(map(to_tensor, [e.state.flatten() for e in experiences if e is not None]))
        actions = list(map(to_tensor, [e.action for e in experiences if e is not None]))
        rewards = list(map(to_tensor, [e.reward for e in experiences if e is not None]))
        next_states = list(map(to_tensor, [e.next_state for e in experiences if e is not None]))
        all_next_states = list(map(to_tensor, [e.next_state.flatten() for e in experiences if e is not None]))
        dones = list(map(to_tensor, [e.done for e in experiences if e is not None]))

        return (states, all_states, actions, rewards, next_states, all_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    buffer = ReplayBuffer(10, torch.device("cpu"), 0)
    buffer.push(np.array([[1, 2, 3]]), [4, 5], [7, 8], [10, 11, 12], [True, False])
    buffer.push(np.array([[1, 2, 3]]), [4, 5], [7, 8], [10, 11, 12], [True, False])
    obs, action, reward, next_obs, done = buffer.sample(2)
    # print(f'obs: {obs.shape}, action: {action.shape}, reward: {reward.shape}, next_obs: {next_obs.shape}, done: {done.shape}')
    print(f'obs: {obs}\n action: {action}\n reward: {reward}\n next_obs: {next_obs}\n done: {done}')
