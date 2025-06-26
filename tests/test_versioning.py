import torch
from src.models.transformer.versioning import ModelVersionManager
from src.models.transformer.transformer_wrapper import TransformerModelWrapper


def test_version_save_load(tmp_path):
    manager = ModelVersionManager(model_dir=tmp_path)
    model = TransformerModelWrapper(seq_length=2, epochs=1)
    model.model = torch.nn.Identity()
    model.is_fitted = True
    vid = manager.save_model(model)
    loaded, meta = manager.load_model(vid)
    assert meta['version'] == vid
    assert loaded.is_fitted
