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
DoubleBufTilePass — CustomPreSchedulingPasses-compatible compiler pass.

Tags tiled LX buffers in the Inductor FX graph with ``double_buffer_slot``
("A" or "B") and ``tile_bytes`` dynamic attributes on their
``LifetimeBoundBuffer`` objects.  These attributes are later consumed by
``DoubleBufTileAllocator`` during LX planning (``_maybe_scratchpad_planning``).

Registration
------------
Add to ``CustomPreSchedulingPasses.__init__`` in ``passes.py``, immediately
**after** ``_distribute_work`` and **before** ``_maybe_scratchpad_planning``::

    from torch_spyre._inductor.scratchpad.double_buf_tile_pass import (
        DoubleBufTilePass,
    )
    ...
    self.passes = [
        ...
        _distribute_work,
        DoubleBufTilePass(),          # ← insert here
        _maybe_scratchpad_planning,
        ...
    ]

Graph interface
---------------
Takes a ``GraphLowering`` object (same type as all other passes in the list).
Reads ``graph.operations`` (topological order, guaranteed by ``GraphLowering``).
For each qualifying operation reads ``node.get_name()`` and
``node.node.layout.size`` to compute the output byte footprint.

Qualification criteria
----------------------
An operation qualifies for double-buffer tagging when:

1. Its output byte footprint >= ``min_tile_bytes`` (default 1 KB).
   Tiny pointwise temporaries are excluded.
2. ``2 * footprint <= lx_capacity_bytes``  (fits the double-buffer budget).
   Tiles too large for LX are left for the standard HBM path.

Confirmed hardware facts
------------------------
- LX memory is 10× faster than HBM (torch-spyre issue #1947, verbatim).
- Usable per-core LX capacity ~1,500 KB (issue #3216 / fix #3266).
"""

from __future__ import annotations

from typing import Any

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.scratchpad.double_buf_tile_allocator import (
    LX_USABLE_CAPACITY_KB,
    _BANK_ATTR,
    _TILE_BYTES_ATTR,
)

__all__ = ["DoubleBufTilePass"]

logger = get_inductor_logger("scratchpad.double_buf_tile_pass")

# Confirmed lower bound: torch-spyre issue #3216 / fix #3266
_DEFAULT_LX_CAPACITY_BYTES = LX_USABLE_CAPACITY_KB * 1024


class DoubleBufTilePass:
    """
    Pre-scheduling pass that tags tiled LX buffers with Bank_A / Bank_B slots.

    Parameters
    ----------
    lx_capacity_bytes:
        Per-core usable LX capacity in bytes.  Defaults to the confirmed
        value from torch-spyre issue #3216 / fix #3266 (1,500 KB).
    dtype_bytes:
        Bytes per element for footprint estimation.  2 = FP16/BF16 (default).
    min_tile_bytes:
        Minimum output footprint (bytes) for a buffer to be considered tiled.
        Buffers smaller than this are left for the standard greedy path.
        Default: 1024 (1 KB).
    """

    def __init__(
        self,
        lx_capacity_bytes: int = _DEFAULT_LX_CAPACITY_BYTES,
        dtype_bytes: int = 2,
        min_tile_bytes: int = 1024,
    ) -> None:
        self.lx_capacity_bytes = lx_capacity_bytes
        self.dtype_bytes = dtype_bytes
        self.min_tile_bytes = min_tile_bytes

    # ------------------------------------------------------------------
    # Pass entry point — called by CustomPreSchedulingPasses.__call__
    # ------------------------------------------------------------------

    def __call__(self, graph: Any) -> None:
        """
        Tag qualifying operations in ``graph.operations`` with
        ``double_buffer_slot`` and ``tile_bytes`` dynamic attributes.

        Operations that do not qualify are left untouched.
        """
        tagged = 0
        for op in graph.operations:
            footprint = self._output_bytes(op)
            if footprint < self.min_tile_bytes:
                continue
            if 2 * footprint > self.lx_capacity_bytes:
                logger.debug(
                    "DoubleBufTilePass: skipping %s — footprint %d B exceeds "
                    "half of LX capacity %d B",
                    op.get_name(),
                    footprint,
                    self.lx_capacity_bytes,
                )
                continue
            # Tag: slot assignment (A/B) is deferred to DoubleBufTileAllocator;
            # set None here so the allocator knows to assign it.
            node = op.node
            try:
                object.__setattr__(node, _BANK_ATTR, None)
                object.__setattr__(node, _TILE_BYTES_ATTR, footprint)
            except (AttributeError, TypeError):
                # node is not a LifetimeBoundBuffer yet at this point —
                # the attribute is carried on the graph node and transferred
                # to the LifetimeBoundBuffer during scratchpad_planning.
                # Store on the op directly as a side-channel for now.
                op.__dict__[_BANK_ATTR] = None
                op.__dict__[_TILE_BYTES_ATTR] = footprint
            tagged += 1

        if tagged:
            logger.debug(
                "DoubleBufTilePass: tagged %d operations for double-buffering",
                tagged,
            )

    # ------------------------------------------------------------------
    # Output footprint estimation
    # ------------------------------------------------------------------

    def _output_bytes(self, op: Any) -> int:
        """
        Return the output tensor byte footprint for ``op``.

        Tries ``op.node.layout.size`` (real Inductor ``ComputedBuffer`` layout,
        a list of sympy/int extents), then falls back to 0.
        """
        try:
            size = op.node.layout.size
            total = 1
            for dim in size:
                total *= int(dim)
            return total * self.dtype_bytes
        except (AttributeError, TypeError, ValueError):
            return 0
