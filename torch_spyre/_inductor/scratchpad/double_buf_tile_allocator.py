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
DoubleBufTileAllocator — Bank_A / Bank_B double-buffer LX placement solver.

Assigns Bank_A / Bank_B slots to tiled LX buffers and places them at
non-overlapping byte addresses within the per-core LX capacity.

Implements the ``MemoryPlanSolver`` interface (``plan_layout()``) and is
registered in ``_PLACEMENT_SOLVERS`` under key ``"double_buf_tile"``,
selectable via ``LAYOUT_SOLVER=double_buf_tile``.

Design
------
Splits LX capacity into two equal banks::

    Bank_A : [0,              lx_capacity // 2)
    Bank_B : [lx_capacity//2, lx_capacity)

Tiled buffers (buffers tagged by ``DoubleBufTilePass`` with the
``double_buffer_slot`` dynamic attribute) are assigned even (Bank_A) or
odd (Bank_B) slots in the order they appear in the buffer list, matching
the ping-pong schedule from IndirectTileScheduler:

    Prologue  : prefetch block 0 → Bank_A
    Step 0    : compute Bank_A  ‖  prefetch block 1 → Bank_B
    Step 1    : compute Bank_B  ‖  prefetch block 2 → Bank_A
    ...

Within each bank non-overlapping buffers share an address (lifetime reuse,
same strategy as ``GreedyLayoutSolver``).  Overlapping buffers are stacked
sequentially.  Non-tiled buffers are packed in the region beyond the two
banks, or by lifetime-reuse within ``[0, lx_capacity)`` when the banks
consume all available LX.

Hardware facts confirmed from the torch-spyre issue tracker
-----------------------------------------------------------
- LX memory is 10× faster than HBM (issue #1947, verbatim).
- Usable per-core LX capacity ~1,500 KB (issue #3216 / fix #3266).
"""

from __future__ import annotations

from collections.abc import Sequence

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)

__all__ = ["DoubleBufTileAllocator"]

logger = get_inductor_logger("scratchpad.double_buf_tile_allocator")

# Dynamic attribute names written onto LifetimeBoundBuffer by DoubleBufTilePass.
# These are not fields on the real class — they are set via object.__setattr__
# so that dataclass frozen-ness (if any) is bypassed cleanly.
_BANK_ATTR = "double_buffer_slot"   # "A" | "B"
_TILE_BYTES_ATTR = "tile_bytes"     # int; 0 = not a tiled buffer


class DoubleBufTileAllocator(MemoryPlanSolver):
    """
    Bank_A / Bank_B double-buffer LX placement solver.

    Registered in ``_PLACEMENT_SOLVERS`` as ``"double_buf_tile"``.
    Activate via the environment variable ``LAYOUT_SOLVER=double_buf_tile``
    or by setting ``config.layout_solver = "double_buf_tile"`` in tests.

    ``supports_paired_buffers = False``: buffers that carry ``paired_with``
    entries are excluded through the base-class ``excluded()`` / ``partition()``
    path and are not placed in LX by this solver.
    """

    supports_paired_buffers = False

    # ------------------------------------------------------------------
    # MemoryPlanSolver abstract method
    # ------------------------------------------------------------------

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        """
        Assign Bank_A / Bank_B addresses to tiled buffers; pack non-tiled
        buffers in remaining LX space.

        Steps
        -----
        1. ``partition()`` — split buffers into placeable / excluded.
        2. Validate: tiled buffers whose ``tile_bytes > bank_bytes`` are
           evicted (address stays None, reason recorded in spill_reasons).
        3. ``_assign_slots()`` — give each tiled buffer a Bank_A / Bank_B slot
           if one is not already set.
        4. Pack Bank_A buffers into ``[0, bank_bytes)``.
        5. Pack Bank_B buffers into ``[bank_bytes, 2*bank_bytes)``.
        6. Pack non-tiled buffers into the remainder, or via lifetime reuse
           inside ``[0, lx_capacity)`` when the banks fill all LX.

        Returns the full ``self.buffers`` list with ``.address`` set on every
        successfully placed buffer.
        """
        placeable, _ = self.partition()
        bank_bytes = self.limit // 2

        tiled: list[LifetimeBoundBuffer] = []
        non_tiled: list[LifetimeBoundBuffer] = []
        for buf in placeable:
            if self._is_tiled(buf):
                tiled.append(buf)
            else:
                non_tiled.append(buf)

        # Step 2: evict tiled buffers that overflow one bank
        valid_tiled: list[LifetimeBoundBuffer] = []
        for buf in tiled:
            tile_bytes = getattr(buf, _TILE_BYTES_ATTR, buf.size)
            if tile_bytes > bank_bytes:
                reason = (
                    f"tile_bytes={tile_bytes} B > bank capacity {bank_bytes} B "
                    f"(half of LX limit {self.limit} B); "
                    f"LX overflow class, see torch-spyre issue #3216"
                )
                logger.warning(
                    "DoubleBufTileAllocator: evicting %r — %s", buf.name, reason
                )
                self.spill_reasons[buf.name] = reason
            else:
                valid_tiled.append(buf)

        # Step 3: assign slots
        self._assign_slots(valid_tiled)

        # Steps 4 & 5: pack each bank
        self._pack_region(
            [b for b in valid_tiled if getattr(b, _BANK_ATTR, None) == "A"],
            base=0,
            capacity=bank_bytes,
        )
        self._pack_region(
            [b for b in valid_tiled if getattr(b, _BANK_ATTR, None) == "B"],
            base=bank_bytes,
            capacity=bank_bytes,
        )

        # Step 6: pack non-tiled buffers
        used_by_banks = 2 * bank_bytes
        remaining = self.limit - used_by_banks
        if remaining > 0:
            self._pack_region(non_tiled, base=used_by_banks, capacity=remaining)
        else:
            # Banks fill all LX — reuse dead tiled slots by lifetime overlap
            self._pack_region(non_tiled, base=0, capacity=self.limit)

        if log_lx_usage:
            placed = sum(1 for b in placeable if b.address is not None)
            logger.debug(
                "DoubleBufTileAllocator: placed %d / %d placeable buffers",
                placed,
                len(placeable),
            )

        return self.buffers

    # ------------------------------------------------------------------
    # Slot assignment: even position → Bank_A, odd position → Bank_B
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tiled(buf: LifetimeBoundBuffer) -> bool:
        """A buffer is tiled if it already has a bank slot or tile_bytes > 0."""
        return (
            getattr(buf, _BANK_ATTR, None) is not None
            or getattr(buf, _TILE_BYTES_ATTR, 0) > 0
        )

    @staticmethod
    def _assign_slots(bufs: list[LifetimeBoundBuffer]) -> None:
        """
        Assign Bank_A / Bank_B to tiled buffers that do not yet have a slot.

        Even position in *bufs* → Bank_A (prologue / even loop steps).
        Odd  position           → Bank_B (odd loop steps).
        """
        pos = 0
        for buf in bufs:
            if getattr(buf, _BANK_ATTR, None) is None:
                object.__setattr__(buf, _BANK_ATTR, "A" if pos % 2 == 0 else "B")
            pos += 1

    # ------------------------------------------------------------------
    # Address packing within a memory region (lifetime reuse)
    # ------------------------------------------------------------------

    def _pack_region(
        self,
        bufs: list[LifetimeBoundBuffer],
        base: int,
        capacity: int,
    ) -> None:
        """
        Assign addresses in ``[base, base + capacity)``.

        Non-overlapping buffers share an address (lifetime reuse — the same
        strategy as ``GreedyLayoutSolver``).  Overlapping buffers are placed
        at sequential, non-overlapping addresses.

        Slot record: ``(address, slot_size_bytes, end_time)``.
        A slot is reusable when ``end_time <= buf.start_time`` and
        ``slot_size >= aligned(buf.size)``.

        Buffers that do not fit are recorded in ``self.spill_reasons``
        with ``address`` left as ``None``.
        """
        _align = self.alignment

        def aligned(n: int) -> int:
            return (n + _align - 1) & ~(_align - 1)

        slots: list[tuple[int, int, int]] = []   # (address, slot_sz, end_time)
        cursor = base

        for buf in bufs:
            sz = aligned(buf.size)
            placed = False
            for idx, (addr, slot_sz, end_t) in enumerate(slots):
                if end_t <= buf.start_time and slot_sz >= sz:
                    buf.address = addr
                    slots[idx] = (addr, slot_sz, buf.end_time)
                    placed = True
                    break
            if not placed:
                if cursor + sz > base + capacity:
                    self.spill_reasons[buf.name] = (
                        f"LX region [{base}, {base + capacity}) exhausted "
                        f"placing {buf.name!r} ({sz} B aligned from {buf.size} B)"
                    )
                    continue
                buf.address = cursor
                slots.append((cursor, sz, buf.end_time))
                cursor += sz
