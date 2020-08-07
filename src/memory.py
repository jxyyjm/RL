from collections import namedtuple, deque
import random
import numpy as np


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Memory:
    """Abstract base class for all implemented memories.

    Do not use this abstract base class directly but instead use one of the concrete memories implemented.

    A memory stores interaction sequences between an agent and one or multiple environments.
    To implement your own memory, you have to implement the following methods:
    """
    def put(self, *args, **kwargs):
        raise NotImplementedError()

    def get(self, *args, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


def unpack(buffer):
    """Returns states, actions, rewards, end_states, and a mask for episode boundaries given buffer."""
    states = [t.state for t in buffer]
    actions = [t.action for t in buffer]
    rewards = [t.reward for t in buffer]
    end_states = [t.next_state for t in buffer]
    not_done_mask = [t.next_state is not None for t in buffer]
    return states, actions, rewards, end_states, not_done_mask


class ExperienceReplay:
    def __init__(self, capacity, steps=1, exclude_boundaries=False):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.steps = steps
        self.exclude_boundaries = exclude_boundaries

    def put(self, transition):
        self.buffer.append(transition)

    def get(self, batch_size):
        buffer = random.sample(self.buffer, batch_size)
        return unpack(buffer)

    def __len__(self):
        return len(self.buffer)

