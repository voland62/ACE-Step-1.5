"""Build the *target* text/lyric conditioning for flow-edit (#1156 PR-B).

Source-side conditioning is already in the payload built by
``preprocess_batch``.  Flow-edit needs a paired *target* condition
(``edit_target_caption`` / ``edit_target_lyrics``); we tokenize and
encode it here using the handler's existing helpers so SFT prompt
formatting, lyric language handling, and padding stay consistent with
the source side.

Split out from ``service_generate_flow_edit.py`` per the 200 LOC cap.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from acestep.constants import DEFAULT_DIT_INSTRUCTION


def _pad_to_batch(values: Optional[List[Any]], default: Any, batch_size: int) -> List[Any]:
    """Right-pad/copy ``values`` to ``batch_size``, falling back to ``default``."""
    out = list(values) if values else [default] * batch_size
    if len(out) < batch_size:
        out = out + [default] * (batch_size - len(out))
    return out


def tokenize_target(
    handler,
    *,
    target_caption: str,
    target_lyrics: str,
    vocal_languages: Optional[List[str]],
    metas: Optional[List[Any]],
    instructions: Optional[List[str]],
    batch_size: int,
):
    """Build padded target text/lyric token tensors via handler helpers.

    Reuses ``_prepare_text_conditioning_inputs`` so the SFT prompt format,
    lyric language formatting, and padding stay consistent with the
    source-side preparation.
    """
    captions = [target_caption] * batch_size
    lyrics = [target_lyrics] * batch_size
    langs = _pad_to_batch(vocal_languages, "unknown", batch_size)
    parsed_metas_list = _pad_to_batch(metas, "", batch_size)
    instr_list = _pad_to_batch(instructions, DEFAULT_DIT_INSTRUCTION, batch_size)

    (
        _text_inputs,
        text_token_idss,
        text_attention_masks,
        lyric_token_idss,
        lyric_attention_masks,
        _nc_text_ids,
        _nc_text_am,
    ) = handler._prepare_text_conditioning_inputs(
        batch_size=batch_size,
        instructions=instr_list,
        captions=captions,
        lyrics=lyrics,
        parsed_metas=parsed_metas_list,
        vocal_languages=langs,
        audio_cover_strength=1.0,  # disable non-cover branch — not needed for edit
    )
    return text_token_idss, text_attention_masks, lyric_token_idss, lyric_attention_masks


def embed_target(
    handler,
    text_token_idss: torch.Tensor,
    lyric_token_idss: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run text + lyric encoders on the target tokens, return embedding tensors.

    Tokens come back from the tokenizer on CPU; the regular batch path
    moves them to ``handler.device`` before encoding (see
    ``preprocess_batch``), so we mirror that here.  Without the move
    text-encoder runs on CUDA / MPS / XPU would hit a device mismatch.
    """
    device = handler.device
    text_token_idss = text_token_idss.to(device=device)
    lyric_token_idss = lyric_token_idss.to(device=device)
    with handler._load_model_context("text_encoder"):
        text_hs = handler.infer_text_embeddings(text_token_idss)
        lyric_hs = handler.infer_lyric_embeddings(lyric_token_idss)
    return text_hs, lyric_hs
