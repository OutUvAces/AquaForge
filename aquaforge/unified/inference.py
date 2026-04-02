"""
AquaForge — **only** supported vessel detection imports for the application.

End-to-end scene proposals: tiled overlap, batch forward, NMS on decoded masks.
Per-spot review/eval: full chip decode into the overlay dict.

There are no alternate backends, hybrid gates, or legacy probability fields on this path.
Import **only** from this module in UI, evaluation, and orchestration code; implementation
details live in :mod:`aquaforge.unified._inference_impl`.
"""

from __future__ import annotations

from aquaforge.unified._inference_impl import (
    run_aquaforge_spot_decode,
    run_aquaforge_tiled_scene_triples,
)

__all__ = ["run_aquaforge_tiled_scene_triples", "run_aquaforge_spot_decode"]
