"""Logging utilities for transformer models."""

import logging
from datetime import datetime
from pathlib import Path


class TransformerLogger:
    def __init__(self, log_dir="logs/transformer"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("transformer")
        self.logger.setLevel(logging.DEBUG)
        self._setup_handlers()

    def _setup_handlers(self):
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(self.log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)
