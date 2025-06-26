"""Performance monitoring utilities."""

import time
from collections import deque
from datetime import datetime
import numpy as np


class PerformanceMonitor:
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds
        self.metrics = deque(maxlen=1000)
        self.alerts = []

    def monitor_prediction(self, X, y=None):
        start = time.time()
        pred = self.model.predict(X)
        latency = time.time() - start
        entry = {'timestamp': datetime.now(), 'latency': latency}
        if y is not None:
            entry['error'] = float(abs(pred - y))
        self.metrics.append(entry)
        self._check()
        return pred

    def _check(self):
        if len(self.metrics) < 100:
            return
        lat = np.mean([m['latency'] for m in self.metrics])
        if lat > self.thresholds.get('max_latency', float('inf')):
            self.alerts.append(('latency', lat))
        errs = [m.get('error') for m in self.metrics if 'error' in m]
        if errs:
            err = np.mean(errs)
            if err > self.thresholds.get('max_error', float('inf')):
                self.alerts.append(('error', err))
