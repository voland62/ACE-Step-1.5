"""Flow-edit dispatch path for ``task_type == "edit"`` (issue #1156).

The regular generation path (``_execute_service_generate_diffusion``)
calls ``model.generate_audio`` with a single set of text + lyric
embeddings.  Flow-edit needs *paired* conditioning (source + target),
so we build a fresh set of target text/lyric embeddings here using the
handler's tokenizer + encoder, then call
``model.flowedit_generate_audio`` with both sets.

Target tokenization + embedding helpers live in
``service_generate_flow_edit_target.py`` (split per the 200 LOC cap).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from loguru import logger

from .service_generate_flow_edit_target import embed_target, tokenize_target


def dispatch_flow_edit(
    handler,
    *,
    payload: Dict[str, Any],
    generate_kwargs: Dict[str, Any],
    seed_param: Any,
    edit_ctx: Dict[str, Any],
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the flow-edit branch and return the same 4-tuple as the regular path.

    Builds target text/lyric embeddings via the handler's tokenizer +
    encoder, runs ``prepare_condition`` on the source side so the
    returned context has the right shape for downstream scoring/LRC,
    then calls ``model.flowedit_generate_audio``.
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

    # Target tokens + embeddings (handler's tokenizer / encoder).
    tar_text_ids, tar_text_am, tar_lyric_ids, tar_lyric_am = tokenize_target(
        handler,
        target_caption=edit_ctx.get("edit_target_caption") or "",
        target_lyrics=edit_ctx.get("edit_target_lyrics") or "",
        vocal_languages=edit_ctx.get("vocal_languages"),
        metas=edit_ctx.get("metas"),
        instructions=edit_ctx.get("instructions"),
        batch_size=bsz,
    )
    tar_text_hs, tar_lyric_hs = embed_target(handler, tar_text_ids, tar_lyric_ids)

    device, dtype = src_latents.device, src_latents.dtype
    tar_text_am = tar_text_am.to(device=device, dtype=dtype)
    tar_lyric_am = tar_lyric_am.to(device=device, dtype=dtype)

    with torch.inference_mode():
        with handler._load_model_context("model"):
            # prepare_condition on the source so the 4-tuple's encoder /
            # context outputs have the post-condition shape downstream
            # auto-LRC / DiT alignment scoring expects.
            attn = torch.ones(
                src_latents.shape[0], src_latents.shape[1],
                device=device, dtype=dtype,
            )
            src_enc_hs, src_enc_am, src_ctx = handler.model.prepare_condition(
                text_hidden_states=payload["text_hidden_states"],
                text_attention_mask=payload["text_attention_mask"],
                lyric_hidden_states=payload["lyric_hidden_states"],
                lyric_attention_mask=payload["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=payload["refer_audio_order_mask"],
                hidden_states=src_latents,
                attention_mask=attn,
                silence_latent=handler.silence_latent,
                src_latents=src_latents,
                chunk_masks=payload["chunk_mask"],
                is_covers=payload["is_covers"],
                precomputed_lm_hints_25Hz=payload.get("precomputed_lm_hints_25Hz"),
            )
            outputs = handler.model.flowedit_generate_audio(
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
                target_text_hidden_states=tar_text_hs,
                target_text_attention_mask=tar_text_am,
                target_lyric_hidden_states=tar_lyric_hs,
                target_lyric_attention_mask=tar_lyric_am,
                seed=seed_param,
                infer_steps=generate_kwargs.get("infer_steps"),
                timesteps=generate_kwargs.get("timesteps"),
                diffusion_guidance_scale=generate_kwargs.get("diffusion_guidance_scale", 1.0),
                cfg_interval_start=generate_kwargs.get("cfg_interval_start", 0.0),
                cfg_interval_end=generate_kwargs.get("cfg_interval_end", 1.0),
                shift=generate_kwargs.get("shift", 1.0),
                velocity_norm_threshold=generate_kwargs.get("velocity_norm_threshold", 0.0),
                velocity_ema_factor=generate_kwargs.get("velocity_ema_factor", 0.0),
                edit_n_min=edit_n_min,
                edit_n_max=edit_n_max,
                edit_n_avg=edit_n_avg,
                precomputed_lm_hints_25Hz=payload.get("precomputed_lm_hints_25Hz"),
                # v1-disabled tricks — pipeline logs them and bypasses.
                sampler_mode=generate_kwargs.get("sampler_mode", "euler"),
                use_adg=generate_kwargs.get("use_adg", False),
                dcw_enabled=generate_kwargs.get("dcw_enabled", False),
            )

    return outputs, src_enc_hs, src_enc_am, src_ctx
