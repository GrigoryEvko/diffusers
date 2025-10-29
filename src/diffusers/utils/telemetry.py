# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Telemetry utilities for tracking performance metrics.

This module provides decorators and utilities for tracking loading performance,
memory usage, and other metrics to help identify optimization opportunities.
"""

import functools
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from . import logging


logger = logging.get_logger(__name__)


@dataclass
class LoadingMetrics:
    """
    Metrics captured during model checkpoint loading.

    Attributes:
        load_time: Total time spent loading in seconds
        memory_peak: Peak memory usage during loading in bytes
        strategy_used: Name of loading strategy used
        file_size: Total size of loaded files in bytes
        num_tensors: Number of tensors loaded
        num_files: Number of files loaded (1 for single file, N for shards)
        device: Target device(s) for loading
        dtype: Data type of loaded tensors
        success: Whether loading completed successfully
        error_message: Error message if loading failed
    """

    load_time: float = 0.0
    memory_peak: int = 0
    strategy_used: str = ""
    file_size: int = 0
    num_tensors: int = 0
    num_files: int = 1
    device: str = "unknown"
    dtype: str = "unknown"
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return asdict(self)

    def __str__(self) -> str:
        """Format metrics as readable string"""
        status = "✓" if self.success else "✗"
        return (
            f"[{status}] {self.strategy_used}: "
            f"{self.load_time:.2f}s, "
            f"{self.num_tensors} tensors, "
            f"{self.file_size / 1e9:.2f}GB, "
            f"peak_mem={self.memory_peak / 1e9:.2f}GB"
        )


def track_loading_performance(func: Callable) -> Callable:
    """
    Decorator to track loading performance metrics.

    Wraps a loading function to automatically track:
    - Loading time
    - Memory usage (GPU if available, else CPU)
    - Success/failure status
    - Error messages

    Usage:
        ```python
        @track_loading_performance
        def load_checkpoint(path):
            return load_state_dict(path)
        ```

    Args:
        func: Function to wrap (should return state_dict or model)

    Returns:
        Wrapped function that tracks metrics
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize metrics
        metrics = LoadingMetrics()
        metrics.strategy_used = kwargs.get("strategy", "unknown")

        # Track device
        if "device_map" in kwargs and kwargs["device_map"]:
            device_map = kwargs["device_map"]
            if isinstance(device_map, str):
                metrics.device = device_map
            elif isinstance(device_map, dict):
                devices = set(device_map.values())
                metrics.device = f"{len(devices)} devices: {list(devices)[:3]}"
        else:
            metrics.device = "cpu"

        # Track dtype
        if "dtype" in kwargs and kwargs["dtype"]:
            metrics.dtype = str(kwargs["dtype"])

        # Track memory before loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0

        # Track time
        start_time = time.time()

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Track success
            metrics.success = True

            # Extract metrics from result
            if isinstance(result, dict):
                metrics.num_tensors = len(result)
                metrics.file_size = sum(
                    t.numel() * t.element_size() for t in result.values() if isinstance(t, torch.Tensor)
                )

        except Exception as e:
            # Track failure
            metrics.success = False
            metrics.error_message = str(e)
            logger.error(f"Loading failed: {e}")
            raise

        finally:
            # Always track time and memory
            metrics.load_time = time.time() - start_time

            if torch.cuda.is_available():
                metrics.memory_peak = torch.cuda.max_memory_allocated() - start_memory
            else:
                metrics.memory_peak = 0

            # Log metrics
            logger.debug(f"Loading metrics: {metrics}")

            # Store metrics in result if possible
            if isinstance(result, dict) and not isinstance(result, torch.Tensor):
                # Add metrics metadata
                result["_loading_metrics"] = metrics

        return result

    return wrapper


class PerformanceTracker:
    """
    Context manager for tracking performance of code blocks.

    Usage:
        ```python
        with PerformanceTracker("loading_phase") as tracker:
            state_dict = load_checkpoint(path)

        print(f"Loading took {tracker.elapsed_time:.2f}s")
        print(f"Peak memory: {tracker.peak_memory / 1e9:.2f}GB")
        ```
    """

    def __init__(self, name: str, track_memory: bool = True):
        """
        Initialize performance tracker.

        Args:
            name: Name of the code block being tracked
            track_memory: Whether to track memory usage (GPU if available)
        """
        self.name = name
        self.track_memory = track_memory
        self.start_time = 0.0
        self.end_time = 0.0
        self.start_memory = 0
        self.peak_memory = 0

    def __enter__(self):
        """Start tracking"""
        self.start_time = time.time()

        if self.track_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()

        logger.debug(f"Starting {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking"""
        self.end_time = time.time()

        if self.track_memory and torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated() - self.start_memory

        elapsed = self.elapsed_time
        if self.track_memory and self.peak_memory > 0:
            logger.debug(f"{self.name} completed in {elapsed:.2f}s, peak memory: {self.peak_memory / 1e9:.2f}GB")
        else:
            logger.debug(f"{self.name} completed in {elapsed:.2f}s")

        return False  # Don't suppress exceptions

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


def get_gpu_memory_stats() -> Dict[str, int]:
    """
    Get current GPU memory statistics.

    Returns:
        Dictionary with memory stats (in bytes):
        - allocated: Currently allocated memory
        - reserved: Reserved memory by caching allocator
        - max_allocated: Peak allocated memory since last reset
        - total: Total GPU memory
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "total": 0}

    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "total": torch.cuda.get_device_properties(0).total_memory,
    }


def log_memory_stats(prefix: str = ""):
    """
    Log current GPU memory statistics.

    Args:
        prefix: Optional prefix for log message
    """
    stats = get_gpu_memory_stats()
    if stats["total"] > 0:
        allocated_gb = stats["allocated"] / 1e9
        total_gb = stats["total"] / 1e9
        utilization = (stats["allocated"] / stats["total"]) * 100

        message = f"{prefix}GPU memory: {allocated_gb:.2f}GB / {total_gb:.2f}GB ({utilization:.1f}%)"
        logger.debug(message)
