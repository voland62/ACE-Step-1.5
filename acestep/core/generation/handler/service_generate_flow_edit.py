"""Flow-edit dispatch path for ``task_type == "edit"`` (issue #1156).

The regular generation path (``_execute_service_generate_diffusion``)
calls ``model.generate_audio`` with a single set of text + lyric
embeddings.  Flow-edit needs *paired* conditioning (source + target),
so we build a fresh set of target text/lyric embeddings here using the
handler's tokenizer + encoder, then call
``model.flowedit_generate_audio`` with both sets.

This module exists to keep ``service_generate_execute.py`` under the
200 LOC cap while still delegating to the handler's existing tokenizer
and encoder helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from acestep.constants import DEFAULT_DIT_INSTRUCTION


def _tokenize_target(
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
    langs = list(vocal_languages) if vocal_languages else ["unknown"] * batch_size
    if len(langs) < batch_size:
        langs = langs + ["unknown"] * (batch_size - len(langs))
    parsed_metas_list = list(metas) if metas else [""] * batch_size
    if len(parsed_metas_list) < batch_size:
        parsed_metas_list = parsed_metas_list + [""] * (batch_size - len(parsed_metas_list))
    instr_list = list(instructions) if instructions else [DEFAULT_DIT_INSTRUCTION] * batch_size
    if len(instr_list) < batch_size:
        instr_list = instr_list + [DEFAULT_DIT_INSTRUCTION] * (batch_size - len(instr_list))

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
        audio_cover_strength=1.0,  # disable non-cover branch — we don't need it for edit
    )
    return text_token_idss, text_attention_masks, lyric_token_idss, lyric_attention_masks


def _embed_target(
    handler,
    text_token_idss: torch.Tensor,
    lyric_token_idss: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run text + lyric encoders on the target tokens, return embedding tensors."""
    with handler._load_model_context("text_encoder"):
        text_hs = handler.infer_text_embeddings(text_token_idss)
        lyric_hs = handler.infer_lyric_embeddings(lyric_token_idss)
    return text_hs, lyric_hs


def dispatch_flow_edit(
    handler,
    *,
    payload: Dict[str, Any],
    generate_kwargs: Dict[str, Any],
    seed_param: Any,
    edit_ctx: Dict[str, Any],
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the flow-edit branch and return the same 4-tuple as the regular path.

    Builds the target text/lyric embeddings inline (no batch_prep
    changes), then calls ``model.flowedit_generate_audio``.  The
    encoder-state outputs returned are the *source* ones so downstream
    metadata persistence is unchanged from the regular path.
    """
    if not hasattr(handler.model, "flowedit_generate_audio"):
        raise RuntimeError(
            "Flow-edit (task_type='edit') requires a base DiT variant — "
            "the loaded model does not expose flowedit_generate_audio. "
            "Supported variants: xl_base, xl_sft, sft, base."
        )
    src_latents = payload["src_latents"]
    bsz = src_latents.shape[0]
    edit_n_min = float(edit_ctx.get("edit_n_min", 0.0))
    edit_n_max = float(edit_ctx.get("edit_n_max", 1.0))
    edit_n_avg = int(edit_ctx.get("edit_n_avg", 1))
    logger.info(
        "[flow_edit] dispatch — task=edit, bsz={}, n_min={}, n_max={}, n_avg={}",
        bsz, edit_n_min, edit_n_max, edit_n_avg,
    )

    # Build target tokens and embeddings using the handler's tokenizer + encoder.
    tar_text_ids, tar_text_am, tar_lyric_ids, tar_lyric_am = _tokenize_target(
        handler,
        target_caption=edit_ctx.get("edit_target_caption") or "",
        target_lyrics=edit_ctx.get("edit_target_lyrics") or "",
        vocal_languages=edit_ctx.get("vocal_languages"),
        metas=edit_ctx.get("metas"),
        instructions=edit_ctx.get("instructions"),
        batch_size=bsz,
    )
    tar_text_hs, tar_lyric_hs = _embed_target(handler, tar_text_ids, tar_lyric_ids)

    # Move tensors to the right device/dtype to match payload conventions.
    device, dtype = src_latents.device, src_latents.dtype
    tar_text_am = tar_text_am.to(device=device, dtype=dtype)
    tar_lyric_am = tar_lyric_am.to(device=device, dtype=dtype)

    with torch.inference_mode():
        with handler._load_model_context("model"):
            outputs = handler.model.flowedit_generate_audio(
                # Source raw inputs — pulled from the prepared payload.
                text_hidden_states=payload["text_hidden_states"],
                text_attention_mask=payload["text_attention_mask"],
                lyric_hidden_states=payload["lyric_hidden_states"],
                lyric_attention_mask=payload["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=payload["refer_audio_order_mask"],
                src_latents=src_latents,
                chunk_masks=payload["chunk_mask"],
                is_covers=payload["is_covers"],
                silence_latent=handler.silence_latent,
                # Target inputs — freshly tokenized and encoded above.
                target_text_hidden_states=tar_text_hs,
                target_text_attention_mask=tar_text_am,
                target_lyric_hidden_states=tar_lyric_hs,
                target_lyric_attention_mask=tar_lyric_am,
                # Sampling
                seed=seed_param,
                infer_steps=generate_kwargs.get("infer_steps"),
                timesteps=generate_kwargs.get("timesteps"),
                diffusion_guidance_scale=generate_kwargs.get("diffusion_guidance_scale", 1.0),
                cfg_interval_start=generate_kwargs.get("cfg_interval_start", 0.0),
                cfg_interval_end=generate_kwargs.get("cfg_interval_end", 1.0),
                shift=generate_kwargs.get("shift", 1.0),
                velocity_norm_threshold=generate_kwargs.get("velocity_norm_threshold", 0.0),
                velocity_ema_factor=generate_kwargs.get("velocity_ema_factor", 0.0),
                # Flow-edit window
                edit_n_min=edit_n_min,
                edit_n_max=edit_n_max,
                edit_n_avg=edit_n_avg,
                # Conditioning hints (forwarded to both prepare_condition calls).
                precomputed_lm_hints_25Hz=payload.get("precomputed_lm_hints_25Hz"),
                # v1-disabled tricks — pipeline logs them and bypasses.
                sampler_mode=generate_kwargs.get("sampler_mode", "euler"),
                use_adg=generate_kwargs.get("use_adg", False),
                dcw_enabled=generate_kwargs.get("dcw_enabled", False),
            )

    # Match the regular path's 4-tuple.  encoder/context tensors are the
    # *source* condition; downstream payload assembly inspects them for
    # latent shapes and intermediate-output stashing only.
    enc_hs = payload["text_hidden_states"]
    enc_am = payload["text_attention_mask"]
    ctx = src_latents
    return outputs, enc_hs, enc_am, ctx
