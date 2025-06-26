"""Online learning utility for transformers."""

import random
from collections import deque
import torch


class OnlineTransformer:
    def __init__(self, model, criterion, optimizer, buffer_size=1000):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.buffer = deque(maxlen=buffer_size)
        self.update_freq = 100

    def add_experience(self, X, y):
        self.buffer.extend(zip(X, y))
        if len(self.buffer) >= self.update_freq:
            self._update()

    def _update(self):
        batch = random.sample(self.buffer, min(32, len(self.buffer)))
        X = torch.stack([torch.tensor(bx, dtype=torch.float32) for bx, _ in batch])
        y = torch.tensor([by for _, by in batch], dtype=torch.float32)
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(X).squeeze()
        loss = self.criterion(preds, y)
        loss.backward()
        self.optimizer.step()
        self.buffer.clear()
