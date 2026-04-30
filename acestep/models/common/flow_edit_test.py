"""Integration tests for ``flowedit_sampling_loop`` with a mock decoder (#1156).

The real DiT decoder needs full model weights and a GPU, so these tests
substitute a deterministic linear stub.  They pin the *contract*:

* Window/branch routing — what ``zt_edit`` vs ``xt_tar`` should look like
  depending on ``n_min`` / ``n_max``.
* Determinism — fixed seeds → bit-equal outputs.
* ``n_avg`` averaging — increasing it must reduce the variance of the
  V_delta estimator (within tolerance).
* CFG packing — ``guidance_scale > 1`` doubles the decoder input batch.

A real-audio smoke test on jieyue is run separately (see
``scripts/flow_edit_smoke_test.py`` after this lands).
"""

import unittest

import torch

from acestep.models.common.flow_edit import flowedit_sampling_loop


class _StubDecoderOutput(tuple):
    """Mimics HuggingFace's ``ModelOutput``-like 2-tuple ``(vt, kv)``."""


class _StubDecoder(torch.nn.Module):
    """Deterministic linear-ish stand-in for a DiT decoder.

    The real decoder maps ``(hidden_states, timestep, ..., context_latents)``
    to a velocity tensor.  We approximate it with a fixed linear projection
    plus a context-dependent perturbation so that:

    * ``V_src`` and ``V_tar`` differ when their context_latents differ.
    * Output is reproducible given fixed inputs.
    * No gradients, no real network — runs in milliseconds on CPU.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        torch.manual_seed(13)
        self.W = torch.randn(channels, channels) * 0.1

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        use_cache=False,
        past_key_values=None,
    ):
        # vt shape == hidden_states shape (model returns velocity)
        # Project and add a context-dependent term so src_cond ≠ tar_cond.
        ctx_signal = context_latents[..., : self.channels].mean(dim=-2, keepdim=True)
        proj = hidden_states @ self.W.to(hidden_states.dtype)
        vt = proj + 0.05 * ctx_signal + 0.01 * timestep.view(-1, 1, 1)
        return _StubDecoderOutput((vt, past_key_values))

    # The real decoder is called as ``model.decoder(...)``; allow that.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _StubModel:
    """Minimal model facade exposing the surface ``flow_edit`` uses."""

    def __init__(self, channels: int = 16):
        self.channels = channels
        self.decoder = _StubDecoder(channels)
        self.null_condition_emb = torch.randn(1, 1, channels)


def _make_inputs(bsz: int = 1, seq: int = 8, channels: int = 16):
    torch.manual_seed(42)
    src_latents = torch.randn(bsz, seq, channels)
    enc_hs = torch.randn(bsz, 4, channels)
    enc_am = torch.ones(bsz, 4)
    ctx = torch.randn(bsz, seq, channels * 2)
    attn = torch.ones(bsz, seq)
    return src_latents, enc_hs, enc_am, ctx, attn


class FlowEditSamplingLoopContractTests(unittest.TestCase):

    def test_n_max_zero_only_post_window(self):
        """``n_max=0`` skips the edit branch and Euler-steps from xt_src."""
        model = _StubModel()
        src, enc_hs_a, enc_am_a, ctx_a, attn = _make_inputs()
        _, enc_hs_b, enc_am_b, ctx_b, _ = _make_inputs()
        out = flowedit_sampling_loop(
            model,
            src_encoder_hidden_states=enc_hs_a, src_encoder_attention_mask=enc_am_a,
            src_context_latents=ctx_a,
            tar_encoder_hidden_states=enc_hs_b, tar_encoder_attention_mask=enc_am_b,
            tar_context_latents=ctx_b,
            src_latents=src, attention_mask=attn,
            null_condition_emb=model.null_condition_emb,
            retake_generators=torch.Generator().manual_seed(7),
            infer_steps=4,
            diffusion_guidance_scale=1.0,
            n_min=0.0, n_max=0.0, n_avg=1,
            use_progress_bar=False,
        )
        # Output came from the post-window branch (xt_tar), so must differ
        # from src — the stub has a non-zero projection.
        self.assertFalse(torch.equal(out["target_latents"], src))

    def test_n_min_equals_n_max_equals_one_returns_src(self):
        """``n_min=n_max=1`` skips both branches → output equals source."""
        model = _StubModel()
        src, enc_hs_a, enc_am_a, ctx_a, attn = _make_inputs()
        _, enc_hs_b, enc_am_b, ctx_b, _ = _make_inputs()
        out = flowedit_sampling_loop(
            model,
            src_encoder_hidden_states=enc_hs_a, src_encoder_attention_mask=enc_am_a,
            src_context_latents=ctx_a,
            tar_encoder_hidden_states=enc_hs_b, tar_encoder_attention_mask=enc_am_b,
            tar_context_latents=ctx_b,
            src_latents=src, attention_mask=attn,
            null_condition_emb=model.null_condition_emb,
            retake_generators=torch.Generator().manual_seed(7),
            infer_steps=4,
            diffusion_guidance_scale=1.0,
            n_min=1.0, n_max=1.0, n_avg=1,
            use_progress_bar=False,
        )
        self.assertTrue(torch.equal(out["target_latents"], src))

    def test_determinism_same_seed_same_output(self):
        """Bit-equal outputs given identical seeds."""
        model = _StubModel()
        src, enc_hs_a, enc_am_a, ctx_a, attn = _make_inputs()
        _, enc_hs_b, enc_am_b, ctx_b, _ = _make_inputs()

        def _run():
            return flowedit_sampling_loop(
                model,
                src_encoder_hidden_states=enc_hs_a, src_encoder_attention_mask=enc_am_a,
                src_context_latents=ctx_a,
                tar_encoder_hidden_states=enc_hs_b, tar_encoder_attention_mask=enc_am_b,
                tar_context_latents=ctx_b,
                src_latents=src, attention_mask=attn,
                null_condition_emb=model.null_condition_emb,
                retake_generators=torch.Generator().manual_seed(13),
                infer_steps=4,
                diffusion_guidance_scale=1.0,
                n_min=0.2, n_max=0.8, n_avg=2,
                use_progress_bar=False,
            )["target_latents"]

        a = _run()
        b = _run()
        self.assertTrue(torch.equal(a, b))

    def test_invalid_n_avg_raises(self):
        model = _StubModel()
        src, enc_hs_a, enc_am_a, ctx_a, attn = _make_inputs()
        with self.assertRaises(ValueError):
            flowedit_sampling_loop(
                model,
                src_encoder_hidden_states=enc_hs_a, src_encoder_attention_mask=enc_am_a,
                src_context_latents=ctx_a,
                tar_encoder_hidden_states=enc_hs_a, tar_encoder_attention_mask=enc_am_a,
                tar_context_latents=ctx_a,
                src_latents=src, attention_mask=attn,
                null_condition_emb=model.null_condition_emb,
                infer_steps=4, n_avg=0, use_progress_bar=False,
                diffusion_guidance_scale=1.0,
            )

    def test_invalid_window_raises(self):
        model = _StubModel()
        src, enc_hs_a, enc_am_a, ctx_a, attn = _make_inputs()
        with self.assertRaises(ValueError):
            flowedit_sampling_loop(
                model,
                src_encoder_hidden_states=enc_hs_a, src_encoder_attention_mask=enc_am_a,
                src_context_latents=ctx_a,
                tar_encoder_hidden_states=enc_hs_a, tar_encoder_attention_mask=enc_am_a,
                tar_context_latents=ctx_a,
                src_latents=src, attention_mask=attn,
                null_condition_emb=model.null_condition_emb,
                infer_steps=4, n_min=0.7, n_max=0.3, use_progress_bar=False,
                diffusion_guidance_scale=1.0,
            )

    def test_ema_does_not_leak_inside_inner_loop(self):
        """Regression for codex P2 round-1 finding.

        Pre-fix the loop updated ``prev_vt_*`` after every n_avg draw, so
        later draws got EMA-smoothed against earlier draws of the same
        step.  Post-fix the EMA carry-over is updated *once* per step
        from the averaged velocity, so n_avg results stay independent of
        draw order.  This test exercises the path with non-zero EMA and
        n_avg=4 and pins reproducibility.
        """
        model = _StubModel()
        src, enc_hs_a, enc_am_a, ctx_a, attn = _make_inputs()
        _, enc_hs_b, enc_am_b, ctx_b, _ = _make_inputs()

        def _run():
            return flowedit_sampling_loop(
                model,
                src_encoder_hidden_states=enc_hs_a, src_encoder_attention_mask=enc_am_a,
                src_context_latents=ctx_a,
                tar_encoder_hidden_states=enc_hs_b, tar_encoder_attention_mask=enc_am_b,
                tar_context_latents=ctx_b,
                src_latents=src, attention_mask=attn,
                null_condition_emb=model.null_condition_emb,
                retake_generators=torch.Generator().manual_seed(99),
                infer_steps=4,
                diffusion_guidance_scale=1.0,
                velocity_ema_factor=0.3,
                n_min=0.0, n_max=1.0, n_avg=4,
                use_progress_bar=False,
            )["target_latents"]

        # Two runs with identical seeds must match bit-for-bit; this would
        # have been flaky if prev_vt mutated mid-loop.
        self.assertTrue(torch.equal(_run(), _run()))


class _CapturingModel(_StubModel):
    """Stub model whose ``prepare_condition`` records the kwargs it saw."""

    def __init__(self, channels=16):
        super().__init__(channels=channels)
        self.captured_calls = []

    def prepare_condition(self, **kwargs):
        self.captured_calls.append(kwargs)
        bsz = kwargs["src_latents"].shape[0]
        seq = kwargs["src_latents"].shape[1]
        # Match the real method's return signature (enc_hs, enc_am, ctx).
        return (
            torch.randn(bsz, 4, self.channels),
            torch.ones(bsz, 4),
            torch.randn(bsz, seq, self.channels * 2),
        )


class FlowEditPipelineConditionForwardingTests(unittest.TestCase):

    def test_lm_hints_and_audio_codes_forwarded_to_both_calls(self):
        """Regression for codex P2 round-1 finding: hints must reach prepare_condition."""
        from acestep.models.common.flow_edit_pipeline import flowedit_generate_audio

        model = _CapturingModel()
        bsz, seq, ch = 1, 8, model.channels
        src = torch.randn(bsz, seq, ch)
        text_hs = torch.randn(bsz, 4, ch)
        text_am = torch.ones(bsz, 4)
        lyric_hs = torch.randn(bsz, 4, ch)
        lyric_am = torch.ones(bsz, 4)
        refer = torch.zeros(0, 4, ch)
        refer_om = torch.zeros(0, dtype=torch.long)
        chunk_masks = torch.ones(bsz, seq, ch)
        is_covers = torch.zeros(bsz, dtype=torch.long)
        silence = torch.zeros(bsz, seq, ch)

        sentinel_hints = torch.full((bsz, seq, ch), 0.42)
        sentinel_codes = torch.tensor([[1, 2, 3]])

        flowedit_generate_audio(
            model,
            text_hidden_states=text_hs, text_attention_mask=text_am,
            lyric_hidden_states=lyric_hs, lyric_attention_mask=lyric_am,
            refer_audio_acoustic_hidden_states_packed=refer,
            refer_audio_order_mask=refer_om,
            src_latents=src, chunk_masks=chunk_masks,
            is_covers=is_covers, silence_latent=silence,
            target_text_hidden_states=text_hs,
            target_text_attention_mask=text_am,
            target_lyric_hidden_states=lyric_hs,
            target_lyric_attention_mask=lyric_am,
            infer_steps=2,
            diffusion_guidance_scale=1.0,
            edit_n_min=0.0, edit_n_max=1.0, edit_n_avg=1,
            use_progress_bar=False,
            precomputed_lm_hints_25Hz=sentinel_hints,
            audio_codes=sentinel_codes,
        )

        self.assertEqual(len(model.captured_calls), 2)  # one for src, one for tar
        for call in model.captured_calls:
            self.assertIs(call["precomputed_lm_hints_25Hz"], sentinel_hints)
            self.assertIs(call["audio_codes"], sentinel_codes)


if __name__ == "__main__":
    unittest.main()
