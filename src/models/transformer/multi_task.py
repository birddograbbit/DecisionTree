"""Simple multi-task transformer wrapper."""

import torch
import torch.nn as nn


class MultiTaskTransformer(nn.Module):
    """Transformer with multiple prediction heads."""

    def __init__(self, base_transformer: nn.Module, task_configs):
        super().__init__()
        self.base = base_transformer
        self.task_heads = nn.ModuleDict({
            name: self._create_head(cfg) for name, cfg in task_configs.items()
        })

    @staticmethod
    def _create_head(cfg):
        return nn.Sequential(
            nn.Linear(cfg["input_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg.get("dropout", 0.1)),
            nn.Linear(cfg["hidden_dim"], cfg["output_dim"]),
        )

    def forward(self, x, task=None):
        features = self.base.get_features(x)
        if task:
            return self.task_heads[task](features)
        return {name: head(features) for name, head in self.task_heads.items()}
