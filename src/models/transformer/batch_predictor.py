import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class BatchPredictor:
    """Efficient batch prediction for large datasets"""
    def __init__(self, model, batch_size=256):
        self.model = model
        self.batch_size = batch_size

    def predict(self, data, device=None):
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size)
        preds = []
        self.model.eval()
        device = device or getattr(self.model, 'device', 'cpu')
        with torch.no_grad():
            for batch, in tqdm(loader, leave=False):
                batch = batch.to(device)
                out = self.model(batch)
                preds.append(out.cpu())
        return torch.cat(preds).numpy()

    def predict_large_dataset(self, dataset, device=None, show_progress=True):
        """Predict using an existing TensorDataset."""
        loader = DataLoader(dataset, batch_size=self.batch_size)
        preds = []
        self.model.eval()
        device = device or getattr(self.model, 'device', 'cpu')
        with torch.no_grad():
            iterator = tqdm(loader, leave=False) if show_progress else loader
            for batch in iterator:
                batch = batch[0].to(device)
                out = self.model(batch)
                preds.append(out.cpu())
        return torch.cat(preds)
