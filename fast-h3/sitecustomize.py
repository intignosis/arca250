"""Raise torch dynamo's recompile limits in every process of this deployment.

Every distinct clip length is a torch.compile shape, and the regional-compile
route runs fullgraph, where exceeding dynamo's recompile limit is a hard
failure (``FailOnRecompileLimitHit``) that kills the engine workers and the
serving process with them. The limit must be raised *inside* the engine
worker processes: FastVideo spawns them (never forks — ``force_spawn``), so
nothing set in the parent carries over, and torch 2.12 maps no environment
variable onto the limit. Worse, the assignment must land *after* FastVideo's
own imports, because it lowers the limits itself at import time
(``layers/lora/linear.py`` sets ``recompile_limit = 16``,
``third_party/longcat_video/.../bsa_interface.py`` sets
``cache_size_limit = 32``).

This file is that mechanism. ``reactor.yaml`` sets ``PYTHONPATH=/app``, so
CPython imports this ``sitecustomize`` at interpreter start in every process
of the container — the runtime and each spawned worker alike. It installs an
import hook that re-raises the limits right after ``torch._dynamo.config``
and after each of FastVideo's two lowering modules execute, so the highest
setting wins no matter the import order. The limit itself comes from
``FASTH3_DYNAMO_RECOMPILE_LIMIT`` (default 64 — 14 legal clip lengths times
the warmed canvases, with room to spare).

Nothing here imports torch: interpreters that never touch dynamo (build
tooling, small subprocesses) pay only for registering the hook.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

# Modules whose import (re)sets dynamo limits; the hook fires after each.
_TARGETS = (
    "torch._dynamo.config",
    "fastvideo.layers.lora.linear",
    "fastvideo.third_party.longcat_video.block_sparse_attention.bsa_interface",
)


def _raise_limits() -> None:
    config = sys.modules.get("torch._dynamo.config")
    if config is None:
        return
    limit = int(os.environ.get("FASTH3_DYNAMO_RECOMPILE_LIMIT", "64"))

    def lift(name: str, floor: int) -> None:
        current = getattr(config, name, None)
        if isinstance(current, int) and current < floor:
            setattr(config, name, floor)

    lift("recompile_limit", limit)
    lift("cache_size_limit", limit)
    # The cross-code-object total; generous so it never binds before the
    # per-object limit does.
    lift("accumulated_recompile_limit", max(512, limit * 8))
    lift("accumulated_cache_size_limit", max(512, limit * 8))
    config.fail_on_recompile_limit_hit = False


class _AfterExecLoader(importlib.abc.Loader):
    """Run the wrapped loader, then re-raise the limits."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        _raise_limits()


class _LimitFinder(importlib.abc.MetaPathFinder):
    """Wrap the loaders of the target modules; transparent to everything else."""

    _resolving = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS or _LimitFinder._resolving:
            return None
        # Resolving the real spec re-enters the import system (parent
        # packages get imported); the guard keeps that re-entry out of here.
        _LimitFinder._resolving = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _LimitFinder._resolving = False
        if fullname in sys.modules:
            # Importing the parents pulled the target in as a side effect;
            # it is already executed, so patch now and stand aside.
            _raise_limits()
            return None
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AfterExecLoader(spec.loader)
        return spec


sys.meta_path.insert(0, _LimitFinder())
