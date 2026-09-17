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
Constraint rules for indirect tiled memory accesses (Paged KV-Cache / Block Table lookups).
Integrates with WorkDivConstraintContext and CP-SAT formulations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Dict, Any, Optional
import sympy
from sympy import Symbol, Expr


@dataclass
class ConstraintResult:
    blocked: Set[Symbol] = field(default_factory=set)
    pinned: Dict[Symbol, int] = field(default_factory=dict)
    forbidden: Set[Symbol] = field(default_factory=set)
    force_output: Set[Symbol] = field(default_factory=set)


@dataclass
class IndirectTileTensorDep:
    name: str
    shape: tuple[int, ...]
    device_coords: tuple[Any, ...]
    is_indirect_table: bool = False
    is_physical_pool: bool = False


@dataclass
class WorkDivConstraintContext:
    op_name: str
    input_tds: list[IndirectTileTensorDep]
    output_td: IndirectTileTensorDep
    it_space: dict[Symbol, Expr]
    it_space_adjusted: dict[Symbol, Expr]
    stick_vars: list[Symbol] = field(default_factory=list)
    reduction_vars: list[Symbol] = field(default_factory=list)
    committed_splits: dict[Symbol, int] = field(default_factory=dict)


def indirect_tile_access_constraints(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """
    Constraint Rule 6: Indirect Tiled Access (Paged KV-Cache / Block Tables).

    Rules:
    1. Physical Pool Block Dim (e.g. pool_blocks) -> FORBIDDEN from being split across cores (splits=1).
    2. Batch & Head Dims -> FORCE_OUTPUT / PROMOTED to enable parallel multi-core execution across SENCores.
    3. Inner stick dimensions (< 64 bytes) -> BLOCKED from non-stick aligned splits.
    """
    forbidden_syms: Set[Symbol] = set()
    force_output_syms: Set[Symbol] = set()
    blocked_syms: Set[Symbol] = set()

    for td in ctx.input_tds:
        if td.is_physical_pool:
            # The leading physical page pool dimension must never be split across cores
            if len(td.device_coords) > 0:
                lead_coord = td.device_coords[0]
                if isinstance(lead_coord, Symbol):
                    forbidden_syms.add(lead_coord)
                elif hasattr(lead_coord, "free_symbols"):
                    forbidden_syms.update(lead_coord.free_symbols)
        elif td.is_indirect_table:
            # Batch dimension on the block table is promoted to output parallelism
            if len(td.device_coords) > 0:
                batch_coord = td.device_coords[0]
                if isinstance(batch_coord, Symbol):
                    force_output_syms.add(batch_coord)
                elif hasattr(batch_coord, "free_symbols"):
                    force_output_syms.update(batch_coord.free_symbols)

    # Stick alignment constraints on inner dimensions
    for sv in ctx.stick_vars:
        blocked_syms.add(sv)

    return ConstraintResult(
        blocked=blocked_syms,
        pinned={},
        forbidden=forbidden_syms,
        force_output=force_output_syms,
    )
