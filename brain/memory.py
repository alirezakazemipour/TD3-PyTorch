import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, *transition):
        self.buffer.append(Transition(*transition))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.capacity

    def sample(self, size):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)
