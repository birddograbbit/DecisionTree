import matplotlib.pyplot as plt
import torch

class AttentionVisualizer:
    """Visualize attention weights for debugging"""
    def __init__(self, model):
        self.model = model

    def plot(self, attn_weights, title='Attention'):
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(attn_weights, aspect='auto', cmap='viridis')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        return fig
