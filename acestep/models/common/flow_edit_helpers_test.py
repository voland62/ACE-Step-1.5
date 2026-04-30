"""Unit tests for ``flow_edit_helpers`` primitives (#1156)."""

import unittest

import torch

from acestep.models.common.apg_guidance import MomentumBuffer
from acestep.models.common.flow_edit_helpers import (
    apply_cfg_branch,
    apply_velocity_clamp,
    apply_velocity_ema,
    build_timestep_schedule,
    draw_fwd_noise,
    pack_for_cfg,
    restore_and_advance_momentum,
    snapshot_momentum,
)


class BuildTimestepScheduleTests(unittest.TestCase):
    """Pin schedule construction — must match the base-variant inline logic."""

    def test_linspace_default_no_shift(self):
        t = build_timestep_schedule(8, 1.0, None, torch.device("cpu"), torch.float32)
        self.assertEqual(t.shape, (9,))
        self.assertAlmostEqual(t[0].item(), 1.0)
        self.assertAlmostEqual(t[-1].item(), 0.0)
        # Even spacing for shift=1
        diffs = (t[:-1] - t[1:]).tolist()
        self.assertTrue(all(abs(d - diffs[0]) < 1e-6 for d in diffs))

    def test_shift_transform_monotonic_decreasing(self):
        t = build_timestep_schedule(8, 3.0, None, torch.device("cpu"), torch.float32)
        for i in range(len(t) - 1):
            self.assertGreater(t[i].item(), t[i + 1].item() - 1e-6)
        self.assertAlmostEqual(t[0].item(), 1.0)
        self.assertAlmostEqual(t[-1].item(), 0.0)

    def test_user_timesteps_passed_through(self):
        custom = torch.tensor([1.0, 0.7, 0.3, 0.0])
        t = build_timestep_schedule(0, 1.0, custom, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.allclose(t, custom))

    def test_user_timesteps_padded_to_zero(self):
        custom = torch.tensor([0.97, 0.5, 0.1])
        t = build_timestep_schedule(0, 1.0, custom, torch.device("cpu"), torch.float32)
        self.assertEqual(t.shape, (4,))
        self.assertAlmostEqual(t[-1].item(), 0.0)


class ApplyCfgBranchTests(unittest.TestCase):
    """Verify CFG dispatch matches the inline base-variant logic."""

    def setUp(self):
        torch.manual_seed(0)
        self.pred_cond = torch.randn(2, 4, 8)
        self.pred_null = torch.randn(2, 4, 8)
        self.pred_packed = torch.cat([self.pred_cond, self.pred_null], dim=0)

    def test_no_cfg_passes_through_unchanged(self):
        out = apply_cfg_branch(self.pred_cond, do_cfg=False, apply_cfg_now=True,
                                guidance_scale=7.0, momentum_buffer=MomentumBuffer())
        self.assertTrue(torch.equal(out, self.pred_cond))

    def test_cfg_outside_interval_returns_cond_chunk(self):
        out = apply_cfg_branch(self.pred_packed, do_cfg=True, apply_cfg_now=False,
                                guidance_scale=7.0, momentum_buffer=MomentumBuffer())
        self.assertTrue(torch.equal(out, self.pred_cond))

    def test_cfg_inside_interval_uses_apg(self):
        buf = MomentumBuffer()
        out = apply_cfg_branch(self.pred_packed, do_cfg=True, apply_cfg_now=True,
                                guidance_scale=7.0, momentum_buffer=buf)
        # APG amplifies the orthogonal component, so the result must differ
        # from a no-op pass-through of the cond chunk.
        self.assertFalse(torch.equal(out, self.pred_cond))


class ApplyVelocityClampTests(unittest.TestCase):
    """Velocity-norm clamp must be a no-op when threshold is 0."""

    def test_threshold_zero_is_noop(self):
        torch.manual_seed(0)
        vt = torch.randn(2, 4, 8)
        xt = torch.randn(2, 4, 8)
        self.assertTrue(torch.equal(apply_velocity_clamp(vt, xt, 0.0), vt))

    def test_clamp_reduces_outlier_norms(self):
        vt = torch.full((1, 4, 8), 100.0)
        xt = torch.full((1, 4, 8), 1.0)
        out = apply_velocity_clamp(vt, xt, 2.0)
        out_norm = torch.norm(out, dim=(1, 2)).item()
        xt_norm = torch.norm(xt, dim=(1, 2)).item()
        self.assertAlmostEqual(out_norm, 2.0 * xt_norm, places=2)


class ApplyVelocityEmaTests(unittest.TestCase):
    """EMA smoothing: per-timestep blend, not per-MC-draw (regression for codex P2)."""

    def test_ema_no_prev_returns_vt(self):
        vt = torch.randn(1, 4, 8)
        self.assertTrue(torch.equal(apply_velocity_ema(vt, None, 0.5), vt))

    def test_ema_zero_factor_returns_vt(self):
        vt = torch.randn(1, 4, 8)
        prev = torch.randn(1, 4, 8)
        self.assertTrue(torch.equal(apply_velocity_ema(vt, prev, 0.0), vt))

    def test_ema_with_prev_blends(self):
        vt = torch.ones(1, 4, 8)
        prev = torch.zeros(1, 4, 8)
        out = apply_velocity_ema(vt, prev, 0.3)
        # 0.7*1 + 0.3*0 = 0.7
        self.assertTrue(torch.allclose(out, torch.full_like(vt, 0.7)))


class MomentumSnapshotTests(unittest.TestCase):
    """Snapshot/restore must isolate APG momentum across MC draws."""

    def test_snapshot_then_advance_matches_single_update(self):
        """One pre-step snapshot + one final update == one direct update."""
        buf_a, buf_b = MomentumBuffer(), MomentumBuffer()
        buf_a.update(torch.tensor([1.0, 2.0]))
        # Snapshot AFTER the prev-step state.
        snap_a, snap_b = snapshot_momentum(buf_a, buf_b)
        # Simulate inner loop "polluting" buf_a with a diff that should NOT
        # carry forward — we restore-and-advance with avg_diff outside.
        buf_a.update(torch.tensor([99.0, 99.0]))
        # Reference: apply only the avg_diff once from the snapshot.
        ref = MomentumBuffer()
        ref.update(torch.tensor([1.0, 2.0]))
        ref.update(torch.tensor([3.0, 4.0]))
        # Restore and advance once with the avg.
        restore_and_advance_momentum(
            buf_a, buf_b, snap_a, snap_b,
            avg_diff_src=torch.tensor([3.0, 4.0]),
            avg_diff_tar=torch.tensor([5.0, 6.0]),
        )
        self.assertTrue(torch.equal(buf_a.running_average, ref.running_average))

    def test_no_cfg_diff_just_rolls_back(self):
        """When CFG is inactive (avg_diff is None) the buffer rolls back."""
        buf_a, buf_b = MomentumBuffer(), MomentumBuffer()
        buf_a.update(torch.tensor([1.0]))
        snap_a, snap_b = snapshot_momentum(buf_a, buf_b)
        # Simulate pollution.
        buf_a.update(torch.tensor([42.0]))
        restore_and_advance_momentum(buf_a, buf_b, snap_a, snap_b, None, None)
        self.assertTrue(torch.equal(buf_a.running_average, snap_a))


class DrawFwdNoiseTests(unittest.TestCase):
    """Generator-aware noise draw must respect both single and per-sample seeds."""

    SHAPE = (3, 4, 8)

    def test_no_generator_returns_random(self):
        a = draw_fwd_noise(self.SHAPE, None, torch.device("cpu"), torch.float32)
        b = draw_fwd_noise(self.SHAPE, None, torch.device("cpu"), torch.float32)
        self.assertEqual(a.shape, self.SHAPE)
        self.assertFalse(torch.equal(a, b))  # different draws

    def test_single_generator_reproducible(self):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        a = draw_fwd_noise(self.SHAPE, g1, torch.device("cpu"), torch.float32)
        b = draw_fwd_noise(self.SHAPE, g2, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(a, b))

    def test_per_sample_generator_list(self):
        gens = [torch.Generator().manual_seed(i) for i in (1, 2, 3)]
        out = draw_fwd_noise(self.SHAPE, gens, torch.device("cpu"), torch.float32)
        self.assertEqual(out.shape, self.SHAPE)
        # Each sample should differ (different seeds).
        self.assertFalse(torch.equal(out[0], out[1]))
        self.assertFalse(torch.equal(out[1], out[2]))


class PackForCfgTests(unittest.TestCase):
    """CFG packing must double tensors and inject the null embedding."""

    def setUp(self):
        torch.manual_seed(0)
        self.enc_hs = torch.randn(2, 16, 32)
        self.enc_am = torch.ones(2, 16)
        self.ctx = torch.randn(2, 24, 64)
        self.attn = torch.ones(2, 24)
        self.null = torch.randn(1, 1, 32)

    def test_no_cfg_passes_through(self):
        out = pack_for_cfg(self.enc_hs, self.enc_am, self.ctx, self.attn, self.null, do_cfg=False)
        self.assertTrue(torch.equal(out[0], self.enc_hs))
        self.assertTrue(torch.equal(out[1], self.enc_am))
        self.assertTrue(torch.equal(out[2], self.ctx))
        self.assertTrue(torch.equal(out[3], self.attn))

    def test_cfg_doubles_batch_and_uses_null(self):
        out = pack_for_cfg(self.enc_hs, self.enc_am, self.ctx, self.attn, self.null, do_cfg=True)
        # Batch dim doubled
        self.assertEqual(out[0].shape[0], 4)
        self.assertEqual(out[1].shape[0], 4)
        self.assertEqual(out[2].shape[0], 4)
        self.assertEqual(out[3].shape[0], 4)
        # First half = original, second half = null-broadcast
        self.assertTrue(torch.equal(out[0][:2], self.enc_hs))
        self.assertTrue(torch.equal(out[0][2:], self.null.expand_as(self.enc_hs)))


if __name__ == "__main__":
    unittest.main()
