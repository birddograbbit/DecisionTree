"""Error recovery helper for transformer models."""

from pathlib import Path
import torch


class RobustTransformer:
    def __init__(self, model, checkpoint_dir="checkpoints"):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_path = None

    def save_checkpoint(self, epoch):
        path = self.checkpoint_dir / f"ckpt_{epoch}.pt"
        torch.save(self.model.state_dict(), path)
        self.last_path = path

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
        self.last_path = path

    def predict_with_fallback(self, data, fallback):
        try:
            return self.model(data)
        except Exception:
            if self.last_path:
                self.load_checkpoint(self.last_path)
            return fallback(data)
