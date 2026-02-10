#!/usr/bin/env python3
"""
Memory Bandwidth Profiler for SGLang rmsnorm Kernel.

Profiles the rmsnorm (RMSNormKernel) kernel with L2 cache clearing
to get accurate memory bandwidth measurements. The kernel applies
RMS normalization with learned weight scaling.

Usage:
    # Basic run with default params (4096 tokens, 4096 hidden dim)
    python scripts/profiling/profile_rmsnorm_bandwidth.py

    # Custom parameters
    python scripts/profiling/profile_rmsnorm_bandwidth.py --tokens 8192 --hidden-dim 8192

    # Without L2 cache clearing (for comparison)
    python scripts/profiling/profile_rmsnorm_bandwidth.py --no-clear-cache

    # With nsys profiling
    nsys profile -o rmsnorm python scripts/profiling/profile_rmsnorm_bandwidth.py
"""

import argparse
import statistics
from dataclasses import dataclass

import torch

# GPU peak bandwidth dictionary (GB/s)
GPU_PEAK_BANDWIDTH = {
    "B200": 8000,  # ~8 TB/s HBM3e
    "H100": 3350,  # 3.35 TB/s HBM3
    "H200": 4800,  # 4.8 TB/s HBM3e
    "A100": 2039,  # 2.0 TB/s HBM2e (80GB variant)
}


# --- L2 Cache Clearing ---


def clear_l2_cache(size_mb: int = 128):
    """Clear L2 cache by allocating and memset'ing a large buffer.

    This technique is from NVIDIA nvbench. By allocating a buffer larger than
    the L2 cache and filling it with zeros, we evict any cached data from
    previous kernel runs, ensuring the next kernel starts with a cold cache.

    Args:
        size_mb: Size of scratch buffer in MB. B200 has ~96MB L2, so 128MB
            is used by default to ensure full eviction.
    """
    size_bytes = size_mb * 1024 * 1024
    scratch = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
    scratch.zero_()  # cudaMemset fills L2 with this data, evicting old cached data
    torch.cuda.synchronize()
    del scratch


# --- Bandwidth Calculation ---


@dataclass
class RMSNormBytes:
    """Memory traffic breakdown for rmsnorm kernel.

    The rmsnorm kernel:
    - Reads: input tensor [tokens, hidden_dim] bf16
    - Reads: weight tensor [hidden_dim] bf16 (typically L2 cached)
    - Writes: output tensor [tokens, hidden_dim] bf16

    Operation: output = (input / rms(input)) * weight
    """

    input_bytes: int  # tokens x hidden_dim x 2 (read bf16 input)
    weight_bytes: int  # hidden_dim x 2 (read bf16 weight, L2 cached)
    output_bytes: int  # tokens x hidden_dim x 2 (write bf16 output)

    @property
    def total_read_bytes(self) -> int:
        return self.input_bytes + self.weight_bytes

    @property
    def total_write_bytes(self) -> int:
        return self.output_bytes

    @property
    def total_bytes(self) -> int:
        return self.total_read_bytes + self.total_write_bytes


def calculate_rmsnorm_bytes(
    tokens: int, hidden_dim: int, dtype_size: int = 2
) -> RMSNormBytes:
    """Calculate memory traffic for rmsnorm kernel.

    Args:
        tokens: Number of tokens
        hidden_dim: Hidden dimension
        dtype_size: Size of element in bytes (2 for bf16)

    Returns:
        RMSNormBytes with detailed memory traffic breakdown
    """
    # Input: [tokens, hidden_dim] bf16
    input_bytes = tokens * hidden_dim * dtype_size

    # Weight: [hidden_dim] bf16 (read once, typically L2 cached for large token counts)
    weight_bytes = hidden_dim * dtype_size

    # Output: [tokens, hidden_dim] bf16
    output_bytes = tokens * hidden_dim * dtype_size

    return RMSNormBytes(
        input_bytes=input_bytes,
        weight_bytes=weight_bytes,
        output_bytes=output_bytes,
    )


# --- Single Kernel Profile ---


@dataclass
class ProfileResult:
    """Results from kernel profiling."""

    times_ms: list[float]
    bytes_info: RMSNormBytes
    peak_gbs: float

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)

    @property
    def bandwidth_gbs(self) -> float:
        """Calculate bandwidth in GB/s from mean time."""
        time_s = self.mean_ms / 1000.0
        return self.bytes_info.total_bytes / time_s / 1e9 if time_s > 0 else 0.0

    @property
    def efficiency_pct(self) -> float:
        """Bandwidth efficiency as percentage of peak."""
        return (self.bandwidth_gbs / self.peak_gbs * 100) if self.peak_gbs > 0 else 0.0


def profile_rmsnorm_kernel(
    tokens: int,
    hidden_dim: int,
    eps: float = 1e-6,
    warmup_iters: int = 5,
    profile_iters: int = 100,
    clear_cache: bool = True,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
    dtype: torch.dtype = torch.bfloat16,
) -> ProfileResult:
    """Profile the rmsnorm kernel.

    Args:
        tokens: Number of tokens
        hidden_dim: Hidden dimension
        eps: Epsilon for numerical stability
        warmup_iters: Number of warmup iterations (no timing)
        profile_iters: Number of profiled iterations
        clear_cache: Whether to clear L2 cache before each iteration
        peak_gbs: Peak GPU bandwidth for efficiency calculation
        dtype: Data type for tensors (bf16 or fp16)

    Returns:
        ProfileResult with timing statistics and bandwidth calculations
    """
    from sgl_kernel import rmsnorm

    dtype_size = torch.finfo(dtype).bits // 8

    # Allocate tensors
    input_tensor = torch.randn(tokens, hidden_dim, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")

    # Calculate expected bytes
    bytes_info = calculate_rmsnorm_bytes(
        tokens=tokens, hidden_dim=hidden_dim, dtype_size=dtype_size
    )

    # Warmup (without L2 clearing to let caches warm up)
    for _ in range(warmup_iters):
        _ = rmsnorm(input_tensor, weight, eps=eps)
    torch.cuda.synchronize()

    # Profile with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for _ in range(profile_iters):
        if clear_cache:
            clear_l2_cache()

        start.record()
        _ = rmsnorm(input_tensor, weight, eps=eps)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        times_ms.append(elapsed_ms)

    return ProfileResult(times_ms=times_ms, bytes_info=bytes_info, peak_gbs=peak_gbs)


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.2f} MB"
    elif num_bytes >= 1e3:
        return f"{num_bytes / 1e3:.2f} KB"
    else:
        return f"{num_bytes} B"


def print_report(
    result: ProfileResult,
    tokens: int,
    hidden_dim: int,
    clear_cache: bool,
    profile_iters: int,
    dtype: torch.dtype,
):
    """Print formatted profiling report."""
    dtype_size = torch.finfo(dtype).bits // 8
    cache_status = "L2 cache cleared" if clear_cache else "L2 cache warm"

    print("=" * 74)
    print("rmsnorm Kernel Memory Bandwidth Profile")
    print("=" * 74)
    print()
    print("Configuration:")
    print(f"  Tokens:          {tokens}")
    print(f"  Hidden dim:      {hidden_dim}")
    print(f"  Total elements:  {tokens * hidden_dim:,}")
    print(f"  Data type:       {dtype} ({dtype_size} bytes/element)")
    print()
    print("Tensor Shapes:")
    print(f"  input:   [{tokens}, {hidden_dim}] {dtype}")
    print(f"  weight:  [{hidden_dim}] {dtype}")
    print(f"  output:  [{tokens}, {hidden_dim}] {dtype}")
    print()
    print("Operation: output = (input / rms(input)) * weight")
    print()
    print("Memory Traffic:")
    b = result.bytes_info
    print(
        f"  Read input:      {format_bytes(b.input_bytes):>12} "
        f"(tokens x hidden_dim x {dtype_size})"
    )
    print(
        f"  Read weight:     {format_bytes(b.weight_bytes):>12} "
        f"(hidden_dim x {dtype_size}, L2 cached)"
    )
    print(
        f"  Write output:    {format_bytes(b.output_bytes):>12} "
        f"(tokens x hidden_dim x {dtype_size})"
    )
    print(f"  ------------------------------------")
    print(f"  Total read:      {format_bytes(b.total_read_bytes):>12}")
    print(f"  Total write:     {format_bytes(b.total_write_bytes):>12}")
    print(f"  Total:           {format_bytes(b.total_bytes):>12}")
    print()
    print(f"Timing ({profile_iters} iterations, {cache_status}):")
    print(f"  Mean:  {result.mean_ms:.4f} ms")
    print(f"  Std:   {result.std_ms:.4f} ms")
    print(f"  Min:   {result.min_ms:.4f} ms")
    print(f"  Max:   {result.max_ms:.4f} ms")
    print()
    print("Bandwidth:")
    print(
        f"  {result.bandwidth_gbs:.1f} GB/s "
        f"({result.efficiency_pct:.1f}% of {result.peak_gbs:.0f} GB/s peak)"
    )
    print("=" * 74)


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Profile memory bandwidth for SGLang rmsnorm kernel"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (multiplies tokens)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=4096,
        help="Number of tokens",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Epsilon for numerical stability",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of profiling iterations",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Disable L2 cache clearing (for comparison)",
    )
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        default=GPU_PEAK_BANDWIDTH["B200"],
        help="Peak GPU memory bandwidth in GB/s",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=list(GPU_PEAK_BANDWIDTH.keys()),
        help="Select GPU type for peak bandwidth (overrides --peak-bandwidth)",
    )

    args = parser.parse_args()

    # Override peak bandwidth if GPU specified
    if args.gpu:
        args.peak_bandwidth = GPU_PEAK_BANDWIDTH[args.gpu]

    clear_cache = not args.no_clear_cache

    # Effective tokens = batch_size * tokens
    effective_tokens = args.batch_size * args.tokens

    print(f"Profiling SGLang rmsnorm kernel...")
    print(
        f"  Batch size: {args.batch_size}, Tokens: {args.tokens}, "
        f"Effective tokens: {effective_tokens}"
    )
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Clear L2 cache: {clear_cache}")
    print()

    result = profile_rmsnorm_kernel(
        tokens=effective_tokens,
        hidden_dim=args.hidden_dim,
        eps=args.eps,
        warmup_iters=args.warmup,
        profile_iters=args.iters,
        clear_cache=clear_cache,
        peak_gbs=args.peak_bandwidth,
    )

    print_report(
        result=result,
        tokens=effective_tokens,
        hidden_dim=args.hidden_dim,
        clear_cache=clear_cache,
        profile_iters=args.iters,
        dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    main()
