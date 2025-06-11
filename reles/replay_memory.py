import random
import threading
from collections import namedtuple
import pickle

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.lock = threading.Lock()

    def push(self, *args):
        """Saves a transition."""
        self.lock.acquire()
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        self.lock.release()

    def sample(self, batch_size):
        self.lock.acquire()
        result = random.sample(self.memory, batch_size)
        self.lock.release()
        return result

    def __len__(self):
        return len(self.memory)