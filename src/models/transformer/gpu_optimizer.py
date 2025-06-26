import torch
from torch.cuda.amp import autocast, GradScaler

class GPUOptimizedTransformer:
    """Utility for mixed precision training"""
    def __init__(self, model, criterion, device='cuda'):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            with autocast():
                out = self.model(x)
                loss = self.criterion(out, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        
    def fit(self, dataloader, optimizer, epochs=1):
        """Train for several epochs with mixed precision."""
        for _ in range(epochs):
            self.train_epoch(dataloader, optimizer)
