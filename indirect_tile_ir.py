# Copyright 2025 The Torch-Spyre Authors.
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
Indirect Tiled Memory Access IR Node & Operations.

This module provides first-class representations for non-affine, runtime-indirect
tiled memory lookups (such as Paged KV-Cache block table lookups) in Inductor and Spyre.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
try:
    import torch
    from torch.library import custom_op

    @custom_op("spyre::indirect_tile_load", mutates_args=())
    def indirect_tile_load(
        base_tensor: torch.Tensor,
        block_table: torch.Tensor,
        batch_idx: int,
        block_idx: int,
        tile_shape: List[int],
    ) -> torch.Tensor:
        """
        Custom Op representing a runtime-indirect tiled load.
        """
        physical_block_id = block_table[batch_idx, block_idx]
        return base_tensor[physical_block_id]

    @indirect_tile_load.register_fake
    def indirect_tile_load_fake(
        base_tensor: torch.Tensor,
        block_table: torch.Tensor,
        batch_idx: int,
        block_idx: int,
        tile_shape: List[int],
    ) -> torch.Tensor:
        """Fake tensor inference for Inductor / AOTAutograd graph tracing."""
        return torch.empty(tile_shape, dtype=base_tensor.dtype, device=base_tensor.device)
except ImportError:
    indirect_tile_load = None
    indirect_tile_load_fake = None


@dataclass
class IndirectTileAccessDescriptor:
    """
    Descriptor capturing compilation metadata for an indirect tile memory operation.
    """
    base_tensor_name: str
    block_table_name: str
    logical_coords: Tuple[Any, ...]
    tile_shape: Tuple[int, ...]
    is_read_only: bool = True
    memory_space: str = "HBM"
    scratchpad_target: str = "SPRAM"
    alignment_bytes: int = 64
    double_buffering_enabled: bool = True

    def get_transfer_bytes(self, dtype_bytes: int = 2) -> int:
        """Calculate the total bytes transferred per tile load."""
        size = 1
        for dim in self.tile_shape:
            size *= dim
        return size * dtype_bytes
