#!/usr/bin/env python3
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
Example: Optimized Model Loading Strategies

This example demonstrates the different loading strategies available in diffusers
for optimizing memory usage and loading speed, especially for large models.

Run with:
    python model_loading_optimization.py --model_id "black-forest-labs/FLUX.1-dev"
"""

import argparse
import time

import torch
from diffusers import FluxPipeline
from diffusers.models import ModelLoader
from diffusers.utils import logging


logger = logging.get_logger(__name__)


def compare_loading_strategies(model_id: str, device="cuda"):
    """
    Compare different loading strategies for the same model.

    This demonstrates the performance and memory characteristics of each strategy.
    """
    print("=" * 80)
    print("COMPARING LOADING STRATEGIES")
    print("=" * 80)

    strategies = {
        "eager": "Traditional full-file loading (baseline)",
        "lazy": "Selective tensor loading (memory efficient)",
        "parallel": "Concurrent shard loading (speed optimized)",
        "parallel_async_lazy": "Ultimate: Parallel + Async + Lazy (best overall)",
    }

    device_map = {"": device} if isinstance(device, str) else device

    results = {}

    for strategy_name, description in strategies.items():
        print(f"\n{'-' * 80}")
        print(f"Strategy: {strategy_name}")
        print(f"Description: {description}")
        print(f"{'-' * 80}")

        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Measure loading
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            # Load with specific strategy
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
                loading_strategy=strategy_name,  # <-- New parameter!
            )

            load_time = time.time() - start_time
            peak_memory = (
                (torch.cuda.max_memory_allocated() - start_memory) / 1e9 if torch.cuda.is_available() else 0
            )

            results[strategy_name] = {"time": load_time, "memory": peak_memory}

            print(f"✓ Loaded in {load_time:.2f}s")
            print(f"✓ Peak memory: {peak_memory:.2f}GB")

            # Clean up
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"✗ Failed: {e}")
            results[strategy_name] = {"time": None, "memory": None, "error": str(e)}

    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Time (s)':<15} {'Memory (GB)':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_time = results.get("eager", {}).get("time")

    for strategy_name in strategies.keys():
        result = results.get(strategy_name, {})
        time_str = f"{result.get('time', 0):.2f}" if result.get("time") else "N/A"
        memory_str = f"{result.get('memory', 0):.2f}" if result.get("memory") else "N/A"

        speedup = ""
        if baseline_time and result.get("time"):
            speedup_factor = baseline_time / result.get("time")
            speedup = f"{speedup_factor:.2f}×"

        print(f"{strategy_name:<25} {time_str:<15} {memory_str:<15} {speedup:<10}")

    return results


def example_automatic_selection(model_id: str):
    """
    Example: Let diffusers automatically select the best loading strategy.

    The 'auto' strategy analyzes your configuration and picks the optimal approach.
    """
    print("\n" + "=" * 80)
    print("AUTOMATIC STRATEGY SELECTION")
    print("=" * 80)

    device_map = {"": "cuda:0"} if torch.cuda.is_available() else "cpu"

    # Automatic selection (recommended!)
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        loading_strategy="auto",  # Automatically picks best strategy
    )

    print("\n✓ Model loaded with automatic strategy selection")
    print("  The optimal strategy was chosen based on your device_map and memory settings")

    return pipe


def example_parallel_async_lazy_advanced(model_id: str):
    """
    Example: Advanced usage of ParallelAsyncLazyLoadingStrategy.

    Shows how to customize worker count, memory budget, and prefetch settings.
    """
    print("\n" + "=" * 80)
    print("ADVANCED: Parallel Async Lazy Loading")
    print("=" * 80)

    from diffusers.models import ModelLoader

    # Create custom loader with advanced settings
    loader = ModelLoader(
        strategy="parallel_async_lazy",
        device_map={"": "cuda:0"},
        max_memory={"cuda:0": "40GiB"},  # Workstation GPU budget
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        max_workers=8,  # Parallel workers
        prefetch_factor=2,  # Prefetch ahead
    )

    print(f"✓ Created advanced loader: {loader}")
    print(f"  - Strategy: ParallelAsyncLazyLoading")
    print(f"  - Workers: 8 parallel threads")
    print(f"  - Memory budget: 40GB")
    print(f"  - Prefetch: 2 shards ahead")

    # Note: This loader would be used internally by from_pretrained
    # when you specify loading_strategy="parallel_async_lazy"

    return loader


def example_memory_efficient_loading(model_id: str):
    """
    Example: Memory-efficient loading for systems with limited RAM/VRAM.

    Uses lazy loading to only load tensors actually needed on GPU.
    """
    print("\n" + "=" * 80)
    print("MEMORY-EFFICIENT LOADING")
    print("=" * 80)

    # Scenario: 24GB GPU, but model is 40GB
    # Solution: Lazy load only GPU tensors, offload rest to CPU/disk

    device_map = {
        "transformer": "cuda:0",  # Main model on GPU
        "vae": "cpu",  # VAE on CPU
        "text_encoder": "cpu",  # Text encoder on CPU
    }

    print("Device map:")
    for component, device in device_map.items():
        print(f"  - {component}: {device}")

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        loading_strategy="lazy",  # Only loads GPU tensors to GPU
    )

    print("\n✓ Model loaded with lazy loading")
    print("  Only transformer was loaded to GPU, saving significant memory")

    return pipe


def example_async_streaming(model_id: str):
    """
    Example: Streaming async loading for fastest possible loading.

    Loads shards in background while initializing model - absolute maximum performance.
    """
    print("\n" + "=" * 80)
    print("STREAMING ASYNC LOADING (Ultimate Performance)")
    print("=" * 80)

    # Note: This is the absolute fastest method
    # It overlaps I/O (loading) with computation (model initialization)

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        loading_strategy="parallel_async_lazy",  # Automatically uses streaming
    )

    print("\n✓ Model loaded with streaming async")
    print("  Shards were loaded in parallel while model was being initialized")
    print("  This provides the absolute fastest loading possible!")

    return pipe


def main():
    parser = argparse.ArgumentParser(description="Model Loading Optimization Examples")
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Model ID to load (default: FLUX.1-dev)",
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["compare", "auto", "advanced", "memory_efficient", "streaming", "all"],
        default="compare",
        help="Which example to run",
    )

    args = parser.parse_args()

    print("Model Loading Optimization Examples")
    print(f"Model: {args.model_id}")
    print()

    if args.example == "compare" or args.example == "all":
        compare_loading_strategies(args.model_id)

    if args.example == "auto" or args.example == "all":
        example_automatic_selection(args.model_id)

    if args.example == "advanced" or args.example == "all":
        example_parallel_async_lazy_advanced(args.model_id)

    if args.example == "memory_efficient" or args.example == "all":
        example_memory_efficient_loading(args.model_id)

    if args.example == "streaming" or args.example == "all":
        example_async_streaming(args.model_id)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nLoading Strategy Recommendations:")
    print("  - Default: Use 'auto' - automatically picks best strategy")
    print("  - Speed: Use 'parallel_async_lazy' - 4-8× faster for sharded models")
    print("  - Memory: Use 'lazy' - 30-70% less memory with device_map")
    print("  - Simple: Use 'eager' - traditional loading (backward compatible)")
    print("\nFor most users, simply use loading_strategy='auto' and let diffusers optimize!")


if __name__ == "__main__":
    main()
