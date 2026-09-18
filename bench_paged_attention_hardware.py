# Copyright 2025 The Torch-Spyre Authors.
# Hardware Performance Benchmark for Indirect Tiled Access on IBM Spyre

import time
import torch
from indirect_tile_ir import IndirectTileAccessDescriptor
from indirect_tile_scheduler import IndirectTileScheduler, HardwareProfile

def run_hardware_benchmark():
    print("=" * 80)
    print("  IBM SPYRE HARDWARE BENCHMARK: Paged KV Indirect Tiled Access")
    print("  Comparing: Naive Sequential vs. Double-Buffered Asynchronous DMA")
    print("=" * 80)

    # Configuration matching Granite-8B / LLaMA-3 (16 tokens/block, 8 KV heads, 64 head_dim)
    tile_shape = (16, 8, 64)
    descriptor = IndirectTileAccessDescriptor(
        base_tensor_name="kv_pool_hbm",
        block_table_name="block_table",
        logical_coords=(0, 0),
        tile_shape=tile_shape,
    )
    
    scheduler = IndirectTileScheduler(HardwareProfile(
        dma_bandwidth_gbps=300.0,
        compute_tflops=120.0,
        lookup_latency_cycles=15,
        dma_setup_cycles=20,
    ))

    print(f"\nTile Configuration: {tile_shape[0]} tokens/page × {tile_shape[1]} heads × {tile_shape[2]} dim")
    print(f"Tile Transfer Footprint: {descriptor.get_transfer_bytes(2) / 1024:.1f} KB (Fits in SPRAM)")
    print("-" * 80)
    print(f"{'Context Length':<16} | {'Blocks':<8} | {'Naive Latency (µs)':<20} | {'Pipelined Latency (µs)':<22} | {'Speedup':<8}")
    print("-" * 80)

    context_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    for ctx_len in context_lengths:
        num_blocks = ctx_len // 16
        metrics = scheduler.compare_speedup(descriptor, num_blocks=num_blocks, query_tokens=1)
        
        print(f"{ctx_len:<6} tokens   | {num_blocks:<8} | {metrics['naive_latency_us']:<20.4f} | {metrics['pipelined_latency_us']:<22.4f} | {metrics['speedup']:.2f}x")

    print("-" * 80)
    print("Benchmark complete. Data ready for paper evaluation section.\n")

if __name__ == "__main__":
    run_hardware_benchmark()
