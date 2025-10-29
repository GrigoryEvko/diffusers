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
Centralized model loading utilities with optimized loading strategies.

This module provides the ModelLoader class, which serves as the main entry point
for loading model checkpoints with various optimization strategies (lazy loading,
parallel loading, streaming, etc.).
"""

import os
from typing import Dict, List, Optional, Union

import torch

from ..utils import logging
from .loading_strategies import (
    LOADING_STRATEGIES,
    EagerLoadingStrategy,
    FastSingleFileStrategy,
    LazyLoadingStrategy,
    LoadingStrategy,
    ParallelAsyncLazyLoadingStrategy,
    ParallelLoadingStrategy,
    StreamingLoadingStrategy,
    get_loading_strategy,
)


logger = logging.get_logger(__name__)


class ModelLoader:
    """
    Centralized checkpoint loader with strategy-based optimization.

    ModelLoader provides a unified interface for loading model checkpoints with
    automatic strategy selection based on model size, device configuration, and
    available memory. Supports multiple loading strategies:

    - **eager**: Traditional full-file loading (backward compatible)
    - **lazy**: Selective tensor loading with safe_open (memory efficient)
    - **parallel**: Concurrent shard loading (fast for sharded models)
    - **streaming**: Overlapped I/O and initialization (lowest memory)
    - **auto**: Automatic strategy selection (recommended)

    Example:
        ```python
        # Automatic strategy selection
        loader = ModelLoader(strategy="auto", device_map={"": "cuda:0"})
        state_dict = loader.load_checkpoint("model.safetensors")

        # Explicit lazy loading for memory efficiency
        loader = ModelLoader(strategy="lazy", device_map=device_map)
        state_dict = loader.load_checkpoint("large_model.safetensors")

        # Parallel loading for sharded models
        loader = ModelLoader(strategy="parallel", max_workers=8)
        state_dict = loader.load_shards(shard_files, index_file)
        ```

    Args:
        strategy: Loading strategy ("auto", "eager", "lazy", "parallel", "streaming")
        device_map: Optional device mapping for distributed loading
        max_memory: Optional memory budget for parallel loading
        dtype: Target dtype for loaded tensors
        low_cpu_mem_usage: Whether to minimize CPU memory usage
        **strategy_kwargs: Additional arguments passed to strategy constructor
    """

    def __init__(
        self,
        strategy: Union[str, LoadingStrategy] = "auto",
        device_map: Optional[Union[str, Dict[str, str]]] = None,
        max_memory: Optional[Dict[str, Union[int, str]]] = None,
        dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = True,
        **strategy_kwargs,
    ):
        self.device_map = device_map
        self.max_memory = max_memory
        self.dtype = dtype or torch.float32
        self.low_cpu_mem_usage = low_cpu_mem_usage

        # Resolve strategy
        if isinstance(strategy, LoadingStrategy):
            self.strategy = strategy
            self.strategy_name = strategy.__class__.__name__
        else:
            self.strategy_name = strategy
            self.strategy = self._resolve_strategy(strategy, **strategy_kwargs)

        logger.debug(f"ModelLoader initialized with strategy: {self.strategy_name}")

    def _resolve_strategy(self, strategy: str, **kwargs) -> LoadingStrategy:
        """
        Resolve strategy name to strategy instance with auto-detection.

        Args:
            strategy: Strategy name or "auto"
            **kwargs: Additional strategy arguments

        Returns:
            LoadingStrategy instance
        """
        if strategy == "auto":
            return self._auto_select_strategy(**kwargs)

        return get_loading_strategy(strategy, **kwargs)

    def _auto_select_strategy(self, **kwargs) -> LoadingStrategy:
        """
        Automatically select optimal loading strategy based on configuration.

        Selection logic:
        - If device_map + low_cpu_mem_usage (typical for sharded models) -> ParallelAsyncLazyLoadingStrategy
        - If device_map distributes weights across devices -> ParallelLoadingStrategy
        - If low_cpu_mem_usage and device_map provided -> LazyLoadingStrategy
        - If simple single GPU case -> FastSingleFileStrategy
        - Otherwise -> EagerLoadingStrategy (backward compatible)

        Returns:
            Automatically selected LoadingStrategy instance
        """
        # Check if we should use the ultimate parallel async lazy loading
        # This is optimal for large sharded models with device_map
        if self.low_cpu_mem_usage and self.device_map and isinstance(self.device_map, dict):
            logger.info("Auto-selected ParallelAsyncLazyLoadingStrategy (optimal for large sharded models)")
            max_memory_gb = self._estimate_max_memory_gb()
            return ParallelAsyncLazyLoadingStrategy(max_memory_gb=max_memory_gb, **kwargs)

        # Check if we have a distributed device map (fallback to regular parallel)
        if self.device_map and isinstance(self.device_map, dict) and len(set(self.device_map.values())) > 1:
            logger.info("Auto-selected ParallelLoadingStrategy (distributed device_map detected)")
            max_memory_gb = self._estimate_max_memory_gb()
            return ParallelLoadingStrategy(max_memory_gb=max_memory_gb, **kwargs)

        # Check if we should use lazy loading (single device case)
        if self.low_cpu_mem_usage and self.device_map:
            logger.info("Auto-selected LazyLoadingStrategy (low_cpu_mem_usage + device_map)")
            return LazyLoadingStrategy(**kwargs)

        # Check if we should use fast single-file loading (simple GPU case)
        if self.device_map and isinstance(self.device_map, str) and self.device_map.startswith("cuda"):
            logger.info("Auto-selected FastSingleFileStrategy (single GPU, optimal for SDXL/SD)")
            return FastSingleFileStrategy(**kwargs)

        # Check if CUDA is available and no device_map (simple case)
        if self.device_map is None and torch.cuda.is_available():
            logger.info("Auto-selected FastSingleFileStrategy (CUDA available, single-file optimization)")
            return FastSingleFileStrategy(**kwargs)

        # Default to eager loading (backward compatible)
        logger.info("Auto-selected EagerLoadingStrategy (default/backward compatible)")
        return EagerLoadingStrategy(**kwargs)

    def _estimate_max_memory_gb(self) -> float:
        """
        Estimate available memory for parallel loading.

        Returns:
            Estimated max memory in GB
        """
        if self.max_memory:
            # Use provided max_memory budget
            total_memory = 0
            for device, mem in self.max_memory.items():
                if isinstance(mem, str):
                    # Parse strings like "20GiB"
                    mem_gb = self._parse_memory_string(mem)
                else:
                    mem_gb = mem / 1e9
                total_memory += mem_gb
            return total_memory

        # Estimate from available CUDA memory
        if torch.cuda.is_available():
            total_vram = sum(
                torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())
            )
            return total_vram / 1e9 * 0.8  # Use 80% of available VRAM

        # Fallback for CPU-only
        return 48.0  # Conservative default for workstation systems

    def _parse_memory_string(self, mem_str: str) -> float:
        """Parse memory string like '20GiB' to GB float"""
        mem_str = mem_str.upper().replace(" ", "")
        if "GIB" in mem_str or "GB" in mem_str:
            return float(mem_str.replace("GIB", "").replace("GB", ""))
        elif "MIB" in mem_str or "MB" in mem_str:
            return float(mem_str.replace("MIB", "").replace("MB", "")) / 1024
        elif "TIB" in mem_str or "TB" in mem_str:
            return float(mem_str.replace("TIB", "").replace("TB", "")) * 1024
        else:
            # Assume bytes
            return float(mem_str) / 1e9

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """
        Load a checkpoint file into a state dictionary.

        Args:
            checkpoint_path: Path to the checkpoint file (.safetensors)

        Returns:
            State dictionary mapping parameter names to tensors

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint format is invalid
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path} using {self.strategy_name}")

        try:
            state_dict = self.strategy.load(checkpoint_path, self.device_map, self.dtype)
            logger.info(f"Successfully loaded {len(state_dict)} tensors")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def load_shards(
        self, shard_files: List[str], index_file: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load sharded checkpoint files into a state dictionary.

        Args:
            shard_files: List of shard file paths
            index_file: Optional index file mapping tensors to shards

        Returns:
            Combined state dictionary from all shards

        Raises:
            FileNotFoundError: If any shard file doesn't exist
            ValueError: If shard format is invalid
        """
        # Validate shard files exist
        missing_files = [f for f in shard_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing shard files: {missing_files}")

        logger.info(f"Loading {len(shard_files)} shards using {self.strategy_name}")

        try:
            state_dict = self.strategy.load_shards(shard_files, index_file, self.device_map, self.dtype)
            logger.info(f"Successfully loaded {len(state_dict)} tensors from {len(shard_files)} shards")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load shards: {e}")
            raise

    def load_shards_streaming(
        self, shard_files: List[str], model: torch.nn.Module, index_file: Optional[str] = None
    ) -> None:
        """
        Load sharded checkpoint with streaming (for StreamingLoadingStrategy).

        This method initializes the model in-place while streaming shard files,
        overlapping I/O and computation for optimal performance and memory usage.

        Args:
            shard_files: List of shard file paths
            model: Model instance to initialize (modified in-place)
            index_file: Optional index file mapping tensors to shards

        Raises:
            ValueError: If strategy doesn't support streaming
            FileNotFoundError: If any shard file doesn't exist
        """
        if not isinstance(self.strategy, StreamingLoadingStrategy):
            raise ValueError(
                f"Streaming loading requires StreamingLoadingStrategy, but got {self.strategy_name}. "
                "Create ModelLoader with strategy='streaming'"
            )

        # Validate shard files exist
        missing_files = [f for f in shard_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing shard files: {missing_files}")

        logger.info(f"Streaming load of {len(shard_files)} shards")

        try:
            self.strategy.load_shards_streaming(shard_files, model, self.device_map, self.dtype)
            logger.info("Successfully completed streaming load")
        except Exception as e:
            logger.error(f"Failed during streaming load: {e}")
            raise

    @property
    def supports_streaming(self) -> bool:
        """Check if current strategy supports streaming loading"""
        return isinstance(self.strategy, StreamingLoadingStrategy)

    @property
    def supports_parallel(self) -> bool:
        """Check if current strategy supports parallel loading"""
        return isinstance(self.strategy, (ParallelLoadingStrategy, StreamingLoadingStrategy))

    def __repr__(self) -> str:
        return (
            f"ModelLoader(strategy={self.strategy_name}, "
            f"device_map={self.device_map}, "
            f"dtype={self.dtype}, "
            f"low_cpu_mem_usage={self.low_cpu_mem_usage})"
        )
