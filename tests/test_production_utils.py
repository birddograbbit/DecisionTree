import torch
from src.models.transformer.logging_config import TransformerLogger
from src.models.transformer.versioning import ModelVersionManager
from src.models.transformer.monitoring import PerformanceMonitor
from src.models.transformer.transformer_wrapper import TransformerModelWrapper
from config import TRANSFORMER_CONFIG


def test_logging_creates_file(tmp_path):
    logger = TransformerLogger(log_dir=tmp_path)
    logger.info("test")
    assert any(p.suffix == '.log' for p in tmp_path.iterdir())


def test_version_manager_save_load(tmp_path):
    model = TransformerModelWrapper(**TRANSFORMER_CONFIG['default'])
    model.model = torch.nn.Identity()
    model.is_fitted = True
    manager = ModelVersionManager(model_dir=tmp_path)
    vid = manager.save_model(model, metrics={'loss':0.1})
    loaded, meta = manager.load_model(vid)
    assert meta['version'] == vid


def test_monitor_alert():
    class DummyModel:
        def predict(self, x):
            return 0.0
    monitor = PerformanceMonitor(DummyModel(), {'max_latency':0, 'max_error':0})
    for _ in range(101):
        monitor.monitor_prediction(1, y=1)
    assert monitor.alerts
