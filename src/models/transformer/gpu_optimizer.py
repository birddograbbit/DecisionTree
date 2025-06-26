"""GPU optimization utilities for transformer models."""

import torch
from torch.amp import GradScaler, autocast


class GPUOptimizedTransformer:
    def __init__(self, model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        # Use new syntax for GradScaler
        if self.device == 'cuda':
            self.scaler = GradScaler('cuda')
        else:
            # CPU fallback - scaler won't actually scale
            self.scaler = None

    def train_epoch(self, loader, optimizer):
        """Train for a single epoch over a DataLoader."""
        self.model.train()
        epoch_loss = 0.0
        for data, target in loader:
            loss = self.train_step(data, target, optimizer)
            epoch_loss += loss
        return epoch_loss / len(loader)

    def fit(self, loader, optimizer, epochs=1):
        """Fit the model for a number of epochs."""
        history = []
        for _ in range(epochs):
            epoch_loss = self.train_epoch(loader, optimizer)
            history.append(epoch_loss)
        return history
        
    def train_step(self, data, target, optimizer):
        data = data.to(self.device)
        target = target.to(self.device)
        
        optimizer.zero_grad()
        
        if self.device == 'cuda' and self.scaler is not None:
            # GPU with mixed precision
            with autocast('cuda'):
                output = self.model(data)
                loss = torch.nn.functional.mse_loss(output, target)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # CPU or no mixed precision
            output = self.model(data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        return loss.item()
