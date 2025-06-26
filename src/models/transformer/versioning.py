"""Simple model version management."""

import json
from datetime import datetime
from .transformer_wrapper import TransformerModelWrapper
from pathlib import Path


class ModelVersionManager:
    def __init__(self, model_dir="models/transformer"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.model_dir / "versions.json"
        self.versions = self._load()

    def _load(self):
        if self.version_file.exists():
            with open(self.version_file, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=2)

    def save_model(self, model, metrics=None, tag=None):
        vid = datetime.now().strftime("%Y%m%d%H%M%S")
        path = self.model_dir / vid
        path.mkdir(exist_ok=True)
        model_path = path / "model.pt"
        model.save(model_path)
        meta = {
            "version": vid,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "tag": tag or "untagged",
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        self.versions[vid] = meta
        self._save_registry()
        return vid

    def load_model(self, version):
        path = self.model_dir / version / "model.pt"
        model = TransformerModelWrapper.load(path)
        with open(self.model_dir / version / "metadata.json") as f:
            meta = json.load(f)
        return model, meta
