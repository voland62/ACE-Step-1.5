"""Multi-GPU detection and smart VRAM allocation for model parallelism.

Provides helpers to detect multiple CUDA GPUs and compute an optimal
component→device assignment for ACE-Step inference.  When a single GPU
cannot hold the DiT model, the allocation signals accelerate-based
intra-model splitting.

See ``compute_component_device_map`` for the allocation strategy.
"""

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from acestep.gpu_config import LM_VRAM, MODEL_VRAM, VRAM_SAFETY_MARGIN_GB


# ---------------------------------------------------------------------------
# VRAM helpers shared with gpu_config (constants imported above)
# ---------------------------------------------------------------------------

# Components co-located with DiT (always on the same device)
_DIT_COMPANION_VRAM_GB = (
    MODEL_VRAM["vae"]
    + MODEL_VRAM["text_encoder"]
    + MODEL_VRAM["silence_latent"]
    + MODEL_VRAM["cuda_context"]
)


def _dit_vram_gb(dit_type: str) -> float:
    """Return DiT weight VRAM in GB for the given type key."""
    key = f"dit_{dit_type}"
    return MODEL_VRAM.get(key, MODEL_VRAM["dit_turbo"])


def _lm_vram_gb(lm_model_size: str) -> float:
    """Return LM weight + KV-cache VRAM estimate in GB."""
    entry = LM_VRAM.get(lm_model_size)
    if entry is None:
        return 0.0
    return entry["weights"] + entry["kv_cache_4k"]


# ---------------------------------------------------------------------------
# Multi-GPU detection
# ---------------------------------------------------------------------------


def get_multi_gpu_info() -> List[Tuple[int, str, float]]:
    """Detect all CUDA devices and return their properties.

    Returns:
        List of ``(device_index, device_name, memory_gb)`` tuples.
        Empty list when CUDA is unavailable.
    """
    try:
        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
        result: List[Tuple[int, str, float]] = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            result.append((i, props.name, mem_gb))
        return result
    except Exception as exc:
        logger.warning("Failed to enumerate CUDA devices: {}", exc)
        return []


def get_total_gpu_memory_gb() -> float:
    """Sum VRAM across all available CUDA GPUs.

    Returns:
        Total VRAM in GB, or ``0.0`` when no GPU is available.
    """
    return sum(mem for _, _, mem in get_multi_gpu_info())


def is_multi_gpu_available() -> bool:
    """Return ``True`` when two or more CUDA GPUs are detected."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Smart VRAM allocation
# ---------------------------------------------------------------------------


def needs_dit_intra_model_split(
    per_gpu_memory: List[float],
    dit_type: str,
) -> bool:
    """Return ``True`` when DiT + companions cannot fit on any single GPU.

    Args:
        per_gpu_memory: Available VRAM per GPU in GB.
        dit_type: DiT variant key (``"turbo"``, ``"xl_turbo"``, etc.).
    """
    dit_total = _dit_vram_gb(dit_type) + _DIT_COMPANION_VRAM_GB + VRAM_SAFETY_MARGIN_GB
    return all(gpu_mem < dit_total for gpu_mem in per_gpu_memory)


def compute_component_device_map(
    per_gpu_memory: List[float],
    dit_type: str = "turbo",
    lm_model_size: str = "",
) -> Optional[Dict[str, str]]:
    """Compute optimal component→device assignment across GPUs.

    Returns ``None`` when only one GPU is available (caller should use the
    existing single-GPU path).

    When two or more GPUs are present the strategy is:

    1. If DiT + companions fit on a single GPU, use **inter-model**
       placement: DiT group on the GPU with most free VRAM, LM on the
       next-best GPU.
    2. If DiT cannot fit on any single GPU, signal **intra-model split**
       by setting ``dit`` to ``"auto"`` (caller uses ``accelerate``'s
       ``device_map="auto"``).  LM is placed on the GPU with the most
       remaining VRAM after accounting for the DiT share.

    Args:
        per_gpu_memory: Available VRAM per GPU in GB.
        dit_type: DiT variant key forwarded to ``_dit_vram_gb``.
        lm_model_size: LM size label (``"0.6B"``, ``"1.7B"``, ``"4B"``
            or empty string for no LM).

    Returns:
        Dict mapping component names to device strings, e.g.
        ``{"dit": "cuda:0", "vae": "cuda:0", ...}``, or ``None``
        for single-GPU configurations.
    """
    if len(per_gpu_memory) < 2:
        return None

    dit_weight = _dit_vram_gb(dit_type)
    dit_total = dit_weight + _DIT_COMPANION_VRAM_GB + VRAM_SAFETY_MARGIN_GB
    lm_total = _lm_vram_gb(lm_model_size) + VRAM_SAFETY_MARGIN_GB if lm_model_size else 0.0

    # --- Case 1: DiT fits on one GPU (inter-model placement) ---------------
    if not needs_dit_intra_model_split(per_gpu_memory, dit_type):
        # Pick the GPU with the most VRAM for DiT
        sorted_gpus = sorted(
            enumerate(per_gpu_memory), key=lambda x: x[1], reverse=True
        )
        dit_gpu_idx = sorted_gpus[0][0]
        # LM goes to the other GPU (pick best remaining)
        lm_gpu_idx = sorted_gpus[1][0] if lm_total > 0 else dit_gpu_idx

        device_map = {
            "dit": f"cuda:{dit_gpu_idx}",
            "vae": f"cuda:{dit_gpu_idx}",
            "text_encoder": f"cuda:{dit_gpu_idx}",
            "lm": f"cuda:{lm_gpu_idx}",
        }
        logger.info(
            "[multi-gpu] Inter-model placement: DiT group → cuda:{}, LM → cuda:{}",
            dit_gpu_idx,
            lm_gpu_idx,
        )
        return device_map

    # --- Case 2: DiT needs intra-model split (accelerate dispatch) ---------
    # LM goes to whichever GPU has the most VRAM (accelerate will spread DiT
    # across all GPUs proportionally).
    sorted_gpus = sorted(
        enumerate(per_gpu_memory), key=lambda x: x[1], reverse=True
    )
    lm_gpu_idx = sorted_gpus[0][0] if lm_total > 0 else 0

    device_map = {
        "dit": "auto",  # signal for accelerate intra-model split
        "vae": f"cuda:{sorted_gpus[0][0]}",
        "text_encoder": f"cuda:{sorted_gpus[0][0]}",
        "lm": f"cuda:{lm_gpu_idx}",
    }
    logger.info(
        "[multi-gpu] Intra-model DiT split (accelerate device_map='auto'), "
        "LM → cuda:{}",
        lm_gpu_idx,
    )
    return device_map
