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

import tempfile
from collections.abc import Sequence
from typing import Any
import os
import subprocess
import torch

from torch._inductor.runtime.runtime_utils import cache_dir
import hashlib
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre._inductor.codegen.bundle import generate_bundle
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


def get_output_dir_for_shape(kernel_name: str, specs) -> str:
    """Like get_output_dir() but encodes op shapes in the directory name.

    Fix  each unique (kernel_name, shapes) combination gets its
    own dxp_standalone binary. Two calls with the same kernel but different
    M (e.g. M=2 vs M=3) land in distinct directories and are compiled
    independently, preventing the stale-kernel reuse that left extra rows zero.
    """
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    shape_tag = hashlib.sha256(repr(specs).encode()).hexdigest()[:16]
    prefix = f"{kernel_name}_{shape_tag}_"
    return tempfile.mkdtemp(dir=spyre_dir, prefix=prefix)


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        # Fix 2/3 (#2523): use shape-aware dir so each (kernel, shape)
        # combination gets its own dxp_standalone binary.
        output_dir = get_output_dir_for_shape(kernel_name, specs)
        generate_bundle(kernel_name, output_dir, specs)

        # Invoke backend compiler of SDSC Bundle
        with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
            subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir], check=True)

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
