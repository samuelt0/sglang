#!/usr/bin/env python3
"""
Profile SGLang's prefill stage with NVFP4 quantization.

This script measures latency and memory usage for the prefill stage (forward only,
no sampling) using:
- NVFP4 quantization (modelopt_fp4)
- Configurable input tokens (default: 4096)
- nsys and NVTX for profiling

Usage:
    # Basic profiling (run from repo root)
    python scripts/profile_prefill_nvfp4.py \
        --model-path nvidia/Llama-3.1-8B-Instruct-NVFP4 \
        --input-len 4096

    # With nsys profiling
    nsys profile \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        -o /tmp/prefill_nvfp4_4096 \
        python scripts/profile_prefill_nvfp4.py \
            --model-path nvidia/Llama-3.1-8B-Instruct-NVFP4 \
            --input-len 4096

    # With memory profiling
    python scripts/profile_prefill_nvfp4.py \
        --model-path nvidia/Llama-3.1-8B-Instruct-NVFP4 \
        --input-len 4096 \
        --enable-memory-profile \
        --output-dir /tmp/sglang_profiles

    # Compare different FP4 backends
    for backend in auto flashinfer_cutlass flashinfer_cudnn flashinfer_trtllm; do
        nsys profile \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            -o /tmp/prefill_nvfp4_${backend} \
            python scripts/profile_prefill_nvfp4.py \
                --fp4-gemm-backend ${backend}
    done
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

# Add the python directory to the path for local development
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if os.path.isdir(PYTHON_DIR) and PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger, suppress_other_loggers
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

# Optional imports - handle gracefully if not available
try:
    from sglang.srt.layers.moe import initialize_moe_config
except ImportError:
    initialize_moe_config = None

try:
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
except ImportError:
    initialize_fp4_gemm_config = None

try:
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
except ImportError:
    initialize_fp8_gemm_config = None

try:
    from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
except ImportError:
    PytHooks = None

try:
    from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
    from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
except ImportError:
    prepare_mlp_sync_batch_raw = None
    require_mlp_sync = None
    require_mlp_tp_gather = None


class TreeCacheNamespace(SimpleNamespace):
    """Dummy tree cache for benchmarks (no prefix caching, just allocation)."""

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()


def get_gpu_memory_info():
    """Get GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
    }


def synchronize(device="cuda"):
    """Synchronize the device."""
    torch.get_device_module(device).synchronize()


def clear_kv_pools(model_runner):
    """Clear KV pools, handling version differences."""
    # Clear request-to-token pool
    if hasattr(model_runner, "req_to_token_pool"):
        model_runner.req_to_token_pool.clear()

    # Clear token-to-KV pool (handle different attribute names across versions)
    if hasattr(model_runner, "token_to_kv_pool_allocator"):
        model_runner.token_to_kv_pool_allocator.clear()
    elif hasattr(model_runner, "token_to_kv_pool"):
        model_runner.token_to_kv_pool.clear()


def load_model(server_args, port_args, gpu_id, tp_rank):
    """Load the model runner and tokenizer."""
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Handle ep_size for MoE models
    ep_size = getattr(server_args, "ep_size", 1)
    moe_ep_rank = tp_rank // (server_args.tp_size // ep_size) if ep_size > 0 else 0

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    return model_runner, tokenizer


def prepare_synthetic_inputs(batch_size, input_len):
    """Prepare synthetic input requests for benchmarking."""
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=1,  # We only need prefill, not decode
    )

    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    return reqs


def _maybe_prepare_mlp_sync_batch(batch, model_runner):
    """Prepare MLP sync batch if required (for DP attention)."""
    if prepare_mlp_sync_batch_raw is None or require_mlp_sync is None:
        return

    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


@torch.no_grad()
def prefill_forward_only(reqs, model_runner):
    """
    Run prefill forward pass only (no sampling).

    This is the core profiling target - it measures just the forward pass
    through the model without the sampling overhead.

    Returns:
        logits_output: The logits from the forward pass
        batch: The schedule batch (for potential decode continuation)
    """
    # Get the token_to_kv_pool_allocator, handling version differences
    if hasattr(model_runner, "token_to_kv_pool_allocator"):
        token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
    else:
        token_to_kv_pool_allocator = model_runner.token_to_kv_pool

    # Create dummy tree_cache for benchmarks
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

    # Run forward pass ONLY - no sampling
    logits_output = model_runner.forward(forward_batch).logits_output

    return logits_output, batch


def profile_prefill_only(
    model_path: str,
    input_len: int = 4096,
    batch_size: int = 1,
    fp4_gemm_backend: str = "auto",
    enable_memory_profile: bool = False,
    output_dir: str = "/tmp/sglang_profiles",
    enable_layerwise_nvtx: bool = True,
    tp_size: int = 1,
):
    """
    Profile the prefill stage with NVFP4 quantization.

    Args:
        model_path: Path to the NVFP4 quantized model
        input_len: Number of input tokens (default: 4096)
        batch_size: Batch size (default: 1)
        fp4_gemm_backend: FP4 GEMM backend (auto, flashinfer_cudnn, flashinfer_cutlass, flashinfer_trtllm)
        enable_memory_profile: Enable detailed memory profiling
        output_dir: Output directory for results
        enable_layerwise_nvtx: Enable per-layer NVTX markers
        tp_size: Tensor parallel size
    """
    print(f"Loading model: {model_path}")
    print(f"Quantization: modelopt_fp4, Backend: {fp4_gemm_backend}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure ServerArgs for NVFP4
    server_args = ServerArgs(
        model_path=model_path,
        quantization="modelopt_fp4",
        fp4_gemm_runner_backend=fp4_gemm_backend,
        disable_cuda_graph=True,  # Required for NVTX markers to work properly
        enable_layerwise_nvtx_marker=enable_layerwise_nvtx,
        tp_size=tp_size,
    )

    # Initialize configurations
    _set_envs_and_config(server_args)

    if initialize_moe_config is not None:
        initialize_moe_config(server_args)
    if initialize_fp8_gemm_config is not None:
        initialize_fp8_gemm_config(server_args)
    if initialize_fp4_gemm_config is not None:
        initialize_fp4_gemm_config(server_args)

    # Configure logger
    configure_logger(server_args)
    suppress_other_loggers()

    # Create port args
    port_args = PortArgs.init_new(server_args)

    # Load the model
    gpu_id = 0
    tp_rank = 0
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # Get memory after model load
    mem_after_load = get_gpu_memory_info()
    print(f"Memory after model load: {mem_after_load.get('allocated_gb', 0):.2f} GB")

    # Register NVTX hooks if enabled and available
    if enable_layerwise_nvtx and PytHooks is not None:
        print("Registering layerwise NVTX hooks...")
        nvtx_hooks = PytHooks()
        nvtx_hooks.register_hooks(model_runner.model)
    elif enable_layerwise_nvtx:
        print("Warning: PytHooks not available, skipping layerwise NVTX markers")

    # Prepare synthetic inputs for warmup
    print("Running warmup...")
    clear_kv_pools(model_runner)
    warmup_reqs = prepare_synthetic_inputs(batch_size, min(input_len, 128))
    prefill_forward_only(warmup_reqs, model_runner)
    synchronize("cuda")

    # Prepare inputs for profiling
    print(f"Preparing {input_len} input tokens...")
    reqs = prepare_synthetic_inputs(batch_size, input_len)

    # Clear pools before profiling
    clear_kv_pools(model_runner)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory_info()

    # Enable memory history if requested
    if enable_memory_profile:
        print("Enabling memory profiling...")
        torch.cuda.memory._record_memory_history(max_entries=100000)

    print("\n=== Starting Prefill Profiling (Forward Only) ===")

    # Start CUDA profiler for nsys
    try:
        torch.cuda.cudart().cudaProfilerStart()
        print("CUDA Profiler started (nsys will begin capturing)")
    except Exception as e:
        print(f"Note: CUDA profiler start failed (OK if not using nsys): {e}")

    # Run prefill forward pass ONLY with NVTX markers
    # Use range_push/range_pop for compatibility with all PyTorch versions
    nvtx.range_push(f"prefill_{input_len}_tokens")
    try:
        synchronize("cuda")
        tic = time.perf_counter()
        logits_output, batch = prefill_forward_only(reqs, model_runner)
        synchronize("cuda")
        prefill_latency = time.perf_counter() - tic
    finally:
        nvtx.range_pop()

    # Stop CUDA profiler
    try:
        torch.cuda.cudart().cudaProfilerStop()
        print("CUDA Profiler stopped (nsys should dump traces)")
    except Exception as e:
        print(f"Note: CUDA profiler stop failed (OK if not using nsys): {e}")

    print("=== Profiling Complete ===\n")

    # Get memory after prefill
    mem_after = get_gpu_memory_info()

    # Dump memory snapshot if enabled
    memory_snapshot_path = None
    if enable_memory_profile:
        memory_snapshot_path = os.path.join(
            output_dir, f"memory_prefill_{input_len}_nvfp4.pickle"
        )
        try:
            torch.cuda.memory._dump_snapshot(memory_snapshot_path)
            print(f"Memory snapshot saved to: {memory_snapshot_path}")
        except Exception as e:
            print(f"Failed to dump memory snapshot: {e}")
            memory_snapshot_path = None
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    # Calculate metrics
    throughput = (input_len * batch_size) / prefill_latency
    mem_delta = mem_after.get("allocated_gb", 0) - mem_before.get("allocated_gb", 0)
    peak_mem = mem_after.get("max_allocated_gb", 0)

    # Print results
    print("=" * 60)
    print("PREFILL PROFILING RESULTS (NVFP4) - Forward Only")
    print("=" * 60)
    print(f"Model:               {model_path}")
    print(f"Input tokens:        {input_len}")
    print(f"Batch size:          {batch_size}")
    print(f"FP4 backend:         {fp4_gemm_backend}")
    print(f"Prefill latency:     {prefill_latency:.4f} s")
    print(f"Throughput:          {throughput:.2f} tokens/s")
    print(f"Peak memory:         {peak_mem:.2f} GB")
    print(f"Memory delta:        {mem_delta:.2f} GB")
    print("=" * 60)

    # Save results to JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "quantization": "modelopt_fp4",
        "fp4_gemm_backend": fp4_gemm_backend,
        "input_len": input_len,
        "batch_size": batch_size,
        "prefill_latency_s": prefill_latency,
        "throughput_tokens_per_s": throughput,
        "memory": {
            "before_prefill_gb": mem_before.get("allocated_gb", 0),
            "after_prefill_gb": mem_after.get("allocated_gb", 0),
            "peak_gb": peak_mem,
            "delta_gb": mem_delta,
        },
        "memory_snapshot_path": memory_snapshot_path,
        "note": "forward_only (no sampling overhead)",
    }

    results_path = os.path.join(
        output_dir, f"prefill_results_{input_len}_nvfp4.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile SGLang prefill stage (forward only) with NVFP4 quantization"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/Llama-3.1-8B-Instruct-NVFP4",
        help="Path to the NVFP4 quantized model",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=4096,
        help="Number of input tokens (default: 4096)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--fp4-gemm-backend",
        type=str,
        default="auto",
        choices=["auto", "flashinfer_cudnn", "flashinfer_cutlass", "flashinfer_trtllm"],
        help="FP4 GEMM backend (default: auto)",
    )
    parser.add_argument(
        "--enable-memory-profile",
        action="store_true",
        help="Enable detailed memory profiling with snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/sglang_profiles",
        help="Output directory for results (default: /tmp/sglang_profiles)",
    )
    parser.add_argument(
        "--disable-layerwise-nvtx",
        action="store_true",
        help="Disable per-layer NVTX markers",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    profile_prefill_only(
        model_path=args.model_path,
        input_len=args.input_len,
        batch_size=args.batch_size,
        fp4_gemm_backend=args.fp4_gemm_backend,
        enable_memory_profile=args.enable_memory_profile,
        output_dir=args.output_dir,
        enable_layerwise_nvtx=not args.disable_layerwise_nvtx,
        tp_size=args.tp_size,
    )


if __name__ == "__main__":
    main()
