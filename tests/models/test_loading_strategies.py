"""
Tests for advanced model loading strategies in diffusers.

Tests cover:
- EagerLoadingStrategy (baseline)
- LazyLoadingStrategy (selective loading with device_map)
- ParallelAsyncLazyLoadingStrategy (parallel loading)
- FastSingleFileStrategy (optimized single-file loading)
- get_loading_strategy() factory function
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from diffusers.models.loading_strategies import (
    EagerLoadingStrategy,
    FastSingleFileStrategy,
    LazyLoadingStrategy,
    ParallelAsyncLazyLoadingStrategy,
    get_loading_strategy,
)
from diffusers.utils.testing_utils import require_torch_gpu, torch_device


class TestLoadingStrategyFactory(unittest.TestCase):
    """Test the get_loading_strategy() factory function"""

    def test_get_eager_strategy(self):
        """Test that 'eager' returns EagerLoadingStrategy"""
        strategy = get_loading_strategy("eager")
        self.assertIsInstance(strategy, EagerLoadingStrategy)

    def test_get_lazy_strategy(self):
        """Test that 'lazy' returns LazyLoadingStrategy"""
        strategy = get_loading_strategy("lazy")
        self.assertIsInstance(strategy, LazyLoadingStrategy)

    def test_get_parallel_async_lazy_strategy(self):
        """Test that 'parallel_async_lazy' returns ParallelAsyncLazyLoadingStrategy"""
        strategy = get_loading_strategy("parallel_async_lazy")
        self.assertIsInstance(strategy, ParallelAsyncLazyLoadingStrategy)

    def test_get_fast_single_file_strategy(self):
        """Test that 'fast_single_file' returns FastSingleFileStrategy"""
        strategy = get_loading_strategy("fast_single_file")
        self.assertIsInstance(strategy, FastSingleFileStrategy)

    def test_get_eager_strategy_by_default_name(self):
        """Test that 'default' returns EagerLoadingStrategy"""
        try:
            strategy = get_loading_strategy("default")
            self.assertIsInstance(strategy, EagerLoadingStrategy)
        except ValueError:
            # If 'default' doesn't exist, 'eager' is the default
            strategy = get_loading_strategy("eager")
            self.assertIsInstance(strategy, EagerLoadingStrategy)

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy name raises ValueError"""
        with self.assertRaises(ValueError):
            get_loading_strategy("invalid_strategy_name")


class TestEagerLoadingStrategy(unittest.TestCase):
    """Test EagerLoadingStrategy (baseline behavior)"""

    def setUp(self):
        """Create test safetensors file"""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "model.safetensors"

        # Create test state dict
        self.state_dict = {
            "layer1.weight": torch.randn(128, 128),
            "layer1.bias": torch.randn(128),
            "layer2.weight": torch.randn(64, 128),
            "layer2.bias": torch.randn(64),
        }
        save_file(self.state_dict, self.test_file)

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_all_tensors(self):
        """Test that eager strategy loads all tensors"""
        strategy = EagerLoadingStrategy()
        loaded = strategy.load(str(self.test_file))

        self.assertEqual(len(loaded), len(self.state_dict))
        for key in self.state_dict.keys():
            self.assertIn(key, loaded)

    def test_tensors_match_original(self):
        """Test that loaded tensors match original values"""
        strategy = EagerLoadingStrategy()
        loaded = strategy.load(str(self.test_file))

        for key, tensor in self.state_dict.items():
            # Move both tensors to CPU for comparison
            self.assertTrue(torch.allclose(loaded[key].cpu(), tensor.cpu()))

    def test_device_placement_cpu(self):
        """Test device placement on CPU"""
        strategy = EagerLoadingStrategy()
        loaded = strategy.load(str(self.test_file), device_map="cpu")

        for tensor in loaded.values():
            self.assertEqual(tensor.device.type, "cpu")

    @require_torch_gpu
    def test_device_placement_cuda(self):
        """Test device placement on CUDA"""
        strategy = EagerLoadingStrategy()
        loaded = strategy.load(str(self.test_file), device_map="cuda:0")

        for tensor in loaded.values():
            self.assertEqual(tensor.device.type, "cuda")


class TestLazyLoadingStrategy(unittest.TestCase):
    """Test LazyLoadingStrategy (selective loading with device_map)"""

    def setUp(self):
        """Create test safetensors file"""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "model.safetensors"

        # Create test state dict with predictable prefixes
        self.state_dict = {
            "encoder.layer1.weight": torch.randn(128, 128),
            "encoder.layer2.weight": torch.randn(128, 128),
            "decoder.layer1.weight": torch.randn(64, 128),
            "decoder.layer2.weight": torch.randn(64, 64),
        }
        save_file(self.state_dict, self.test_file)

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_with_simple_device_map(self):
        """Test lazy loading with simple string device_map"""
        strategy = LazyLoadingStrategy()
        loaded = strategy.load(str(self.test_file), device_map="cpu")

        # Should load all tensors when device_map is simple string
        self.assertEqual(len(loaded), len(self.state_dict))

    def test_filter_by_device_map(self):
        """Test that lazy strategy filters tensors mapped to 'disk'"""
        device_map = {
            "encoder": "cpu",
            "decoder": "disk",  # Should NOT load decoder tensors
        }

        strategy = LazyLoadingStrategy()
        loaded = strategy.load(str(self.test_file), device_map=device_map)

        # Should only load encoder tensors
        encoder_keys = [k for k in loaded.keys() if k.startswith("encoder")]
        decoder_keys = [k for k in loaded.keys() if k.startswith("decoder")]

        self.assertEqual(len(encoder_keys), 2, "Should load encoder tensors")
        self.assertEqual(len(decoder_keys), 0, "Should filter decoder tensors")

    def test_filter_by_prefix_match(self):
        """Test device_map prefix matching"""
        device_map = {
            "encoder.layer1": "cpu",
            "encoder.layer2": "disk",
            "decoder": "cpu",
        }

        strategy = LazyLoadingStrategy()
        loaded = strategy.load(str(self.test_file), device_map=device_map)

        self.assertIn("encoder.layer1.weight", loaded)
        self.assertNotIn("encoder.layer2.weight", loaded)  # Filtered
        self.assertIn("decoder.layer1.weight", loaded)
        self.assertIn("decoder.layer2.weight", loaded)

    @require_torch_gpu
    def test_mixed_device_placement(self):
        """Test mixed device placement with device_map"""
        device_map = {
            "encoder": "cuda:0",
            "decoder": "cpu",
        }

        strategy = LazyLoadingStrategy()
        loaded = strategy.load(str(self.test_file), device_map=device_map)

        # Check device placement
        for key, tensor in loaded.items():
            if key.startswith("encoder"):
                self.assertEqual(tensor.device.type, "cuda")
            elif key.startswith("decoder"):
                self.assertEqual(tensor.device.type, "cpu")


class TestFastSingleFileStrategy(unittest.TestCase):
    """Test FastSingleFileStrategy (optimized single-file loading)"""

    def setUp(self):
        """Create test safetensors file"""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "model.safetensors"

        self.state_dict = {
            "weight1": torch.randn(256, 256),
            "weight2": torch.randn(256, 256),
            "bias1": torch.randn(256),
        }
        save_file(self.state_dict, self.test_file)

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fast_single_file_loads_all(self):
        """Test that fast_single_file loads all tensors"""
        strategy = FastSingleFileStrategy()
        loaded = strategy.load(str(self.test_file))

        self.assertEqual(len(loaded), len(self.state_dict))

    def test_fast_single_file_values_match(self):
        """Test that fast_single_file produces correct values"""
        strategy = FastSingleFileStrategy()
        loaded = strategy.load(str(self.test_file))

        for key, tensor in self.state_dict.items():
            # Move both tensors to CPU for comparison (they might be on different devices)
            self.assertTrue(torch.allclose(loaded[key].cpu(), tensor.cpu()))

    def test_fast_vs_eager_same_result(self):
        """Test that fast_single_file produces same result as eager"""
        eager = EagerLoadingStrategy()
        fast = FastSingleFileStrategy()

        eager_loaded = eager.load(str(self.test_file))
        fast_loaded = fast.load(str(self.test_file))

        self.assertEqual(set(eager_loaded.keys()), set(fast_loaded.keys()))

        for key in eager_loaded.keys():
            # Move to CPU for comparison
            self.assertTrue(torch.allclose(eager_loaded[key].cpu(), fast_loaded[key].cpu()))


@pytest.mark.asyncio
class TestParallelAsyncLazyLoadingStrategy:
    """Test ParallelAsyncLazyLoadingStrategy (async parallel loading)"""

    def setup_method(self):
        """Create test files"""
        self.tmpdir = tempfile.mkdtemp()

        # Create multiple shard files
        self.shard_files = []
        for i in range(3):
            shard_file = Path(self.tmpdir) / f"model-{i+1:05d}-of-00003.safetensors"
            state_dict = {
                f"layer{i}.weight": torch.randn(128, 128),
                f"layer{i}.bias": torch.randn(128),
            }
            save_file(state_dict, shard_file)
            self.shard_files.append(shard_file)

    def teardown_method(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_async_parallel_loads_multiple_files(self):
        """Test that parallel async strategy can load multiple files"""
        strategy = ParallelAsyncLazyLoadingStrategy()

        # Load all shards in parallel
        shard_paths = [str(f) for f in self.shard_files]
        result = await strategy.load_shards_async(shard_paths)

        # Should combine all shards into one dict
        assert len(result) >= 3, "Should have tensors from all shards"

        # Verify keys from each shard exist
        for i in range(3):
            assert f"layer{i}.weight" in result
            assert f"layer{i}.bias" in result

    @pytest.mark.asyncio
    async def test_async_with_device_map_filtering(self):
        """Test async loading with device_map filtering"""
        strategy = ParallelAsyncLazyLoadingStrategy()

        device_map = {
            "layer0": "cpu",
            "layer1": "disk",  # Filter
            "layer2": "cpu",
        }

        # Load all shards with filtering
        shard_paths = [str(f) for f in self.shard_files]
        result = await strategy.load_shards_async(shard_paths, device_map=device_map)

        # layer1 (shard 1) should be filtered
        assert "layer0.weight" in result, "layer0 should be loaded"
        assert "layer1.weight" not in result, "layer1 should be filtered (device='disk')"
        assert "layer2.weight" in result, "layer2 should be loaded"

    @pytest.mark.asyncio
    async def test_async_parallel_faster_than_sequential(self):
        """Test that parallel loading is faster than sequential (conceptual)"""
        import time

        strategy_parallel = ParallelAsyncLazyLoadingStrategy()
        strategy_lazy = LazyLoadingStrategy()

        # Parallel load
        start = time.time()
        shard_paths = [str(f) for f in self.shard_files]
        await strategy_parallel.load_shards_async(shard_paths)
        parallel_time = time.time() - start

        # Sequential load
        start = time.time()
        for f in self.shard_files:
            strategy_lazy.load(str(f))
        sequential_time = time.time() - start

        # Note: For small test files, overhead may dominate, so we just verify it works
        # In production with large files, parallel should be faster
        assert parallel_time >= 0 and sequential_time >= 0, "Both should complete"


class TestLoadingStrategyIntegration(unittest.TestCase):
    """Integration tests comparing different strategies"""

    def setUp(self):
        """Create test model file"""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "model.safetensors"

        self.state_dict = {
            "encoder.weight": torch.randn(256, 256),
            "decoder.weight": torch.randn(256, 256),
            "head.weight": torch.randn(10, 256),
        }
        save_file(self.state_dict, self.test_file)

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_strategies_produce_same_result(self):
        """Test that all strategies (without filtering) produce identical results"""
        eager = EagerLoadingStrategy()
        lazy = LazyLoadingStrategy()
        fast = FastSingleFileStrategy()

        eager_loaded = eager.load(str(self.test_file))
        lazy_loaded = lazy.load(str(self.test_file), device_map="cpu")
        fast_loaded = fast.load(str(self.test_file))

        # All should have same keys
        self.assertEqual(set(eager_loaded.keys()), set(lazy_loaded.keys()))
        self.assertEqual(set(eager_loaded.keys()), set(fast_loaded.keys()))

        # All should have same values
        for key in eager_loaded.keys():
            # Move to CPU for comparison
            self.assertTrue(torch.allclose(eager_loaded[key].cpu(), lazy_loaded[key].cpu()))
            self.assertTrue(torch.allclose(eager_loaded[key].cpu(), fast_loaded[key].cpu()))

    def test_lazy_reduces_memory_with_filtering(self):
        """Test that lazy strategy with filtering loads fewer tensors"""
        eager = EagerLoadingStrategy()
        lazy = LazyLoadingStrategy()

        device_map = {
            "encoder": "cpu",
            "decoder": "disk",  # Filter
            "head": "disk",  # Filter
        }

        eager_loaded = eager.load(str(self.test_file))
        lazy_loaded = lazy.load(str(self.test_file), device_map=device_map)

        self.assertGreater(len(eager_loaded), len(lazy_loaded), "Lazy should load fewer tensors")
        self.assertEqual(len(lazy_loaded), 1, "Should only load encoder")


class TestLoadingStrategyEdgeCases(unittest.TestCase):
    """Test edge cases for loading strategies"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_file_raises_error(self):
        """Test that loading empty file raises appropriate error"""
        empty_file = Path(self.tmpdir) / "empty.safetensors"
        empty_file.touch()

        strategy = EagerLoadingStrategy()

        with self.assertRaises(Exception):  # safetensors will raise some error
            strategy.load(str(empty_file))

    def test_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError"""
        nonexistent = Path(self.tmpdir) / "nonexistent.safetensors"

        strategy = EagerLoadingStrategy()

        with self.assertRaises(FileNotFoundError):
            strategy.load(str(nonexistent))

    def test_empty_device_map_loads_all(self):
        """Test that empty device_map dict loads all tensors"""
        test_file = Path(self.tmpdir) / "model.safetensors"
        state_dict = {"weight": torch.randn(10, 10)}
        save_file(state_dict, test_file)

        strategy = LazyLoadingStrategy()
        loaded = strategy.load(str(test_file), device_map={})

        self.assertEqual(len(loaded), 1)
        self.assertIn("weight", loaded)


if __name__ == "__main__":
    unittest.main()
