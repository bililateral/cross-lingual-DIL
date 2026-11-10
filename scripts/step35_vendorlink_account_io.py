"""Reuse qualified role files; only identity checks and bounded runtime IO."""
from pathlib import Path
import shutil
import time

import psutil

from step30_evolution_io import PYTHON, directory_bytes, record, write, verify as pinned
from step35_vendorlink_account_policy import LIMITS


class SharedDiskLimit(RuntimeError):
    """Experiment-wide disk failure, distinct from an individual OOM."""


class Check:
    def __init__(self, root, seconds):
        self.root = Path(root)
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.peak_rss = 0
        self.last_disk = 0
        process = psutil.Process()
        process.cpu_affinity(process.cpu_affinity()[:2])

    def __call__(self):
        now = time.monotonic()
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if now >= self.deadline - 5:
            raise TimeoutError('Worker lifecycle deadline')
        if self.peak_rss > LIMITS['rss_bytes']:
            raise RuntimeError('Worker RSS cap')
        if now - self.last_disk > 5:
            self.last_disk = now
            if (directory_bytes(self.root) > LIMITS['disk_bytes']
                    or shutil.disk_usage(self.root).free < LIMITS['free_disk_reserve']):
                raise SharedDiskLimit('Experiment disk cap/reserve')
