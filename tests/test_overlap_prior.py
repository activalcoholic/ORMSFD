"""CPU checks for ORMSFD overlap supervision, feature fusion, and checkpoints."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ultralytics.nn.modules import OverlapPriorHead, MSFEM
from ultralytics.nn.modules.MSFEM import ChannelSplitPriorFusion, FeatureEnhancementBase
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import IoG
from ultralytics.utils.overlap_prior import batch_overlap_targets, balanced_prior_loss, build_overlap_prior

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def make_batch(height=64, width=96):
    return {
        "img": torch.randn(2, 3, height, width),
        "batch_idx": torch.tensor([0, 0, 1]), "cls": torch.zeros(3, 1),
        "bboxes": torch.tensor([[.4, .5, .4, .4], [.5, .5, .4, .4], [.3, .3, .2, .2]]),
    }


class OverlapPriorTests(unittest.TestCase):
    def test_known_geometry_empty_containment_and_order(self):
            example = torch.tensor([[80, 80, 160, 120], [120, 88, 200, 128]])
            raw = build_overlap_prior(example, (640, 640), 0)
            self.assertAlmostEqual(raw[100, 130].item(), .4, places=6)
            self.assertAlmostEqual(raw.sum().item(), 512, places=3)
            torch.testing.assert_close(raw, build_overlap_prior(example.flip(0), (640, 640), 0))
            for boxes in [torch.empty(0, 4), example[:1], torch.tensor([[0, 0, 5, 5], [10, 10, 15, 15]])]:
                self.assertEqual(build_overlap_prior(boxes, (640, 640)).count_nonzero().item(), 0)
            contained = torch.tensor([[10, 10, 90, 90], [30, 30, 60, 60]])
            self.assertEqual(build_overlap_prior(contained, (100, 100), 0).max().item(), 1)
            self.assertAlmostEqual(max(IoG(example[:1], example[1:]), IoG(example[1:], example[:1])).item(), .4)

    def test_augmented_normalized_coordinates_and_empty_image(self):
            batch = make_batch(64, 96)
            prior = batch_overlap_targets(batch, (8, 12), "cpu", sigma=0)
            boxes = torch.tensor([[.2 * 96, .3 * 64, .6 * 96, .7 * 64],
                                  [.3 * 96, .3 * 64, .7 * 96, .7 * 64]])
            expected = torch.nn.functional.avg_pool2d(build_overlap_prior(boxes, (64, 96), 0)[None, None], 8)
            torch.testing.assert_close(prior[:1], expected, atol=1e-6, rtol=1e-5)
            self.assertEqual(prior[1].count_nonzero().item(), 0)

    def test_soft_bce_and_negative_images_have_useful_gradients(self):
            for kind in ("bce", "l1"):
                logits = torch.full((2, 1, 8, 8), -2.0, requires_grad=True)
                target = torch.zeros_like(logits)
                target[0, 0, 4, 4] = .8
                loss = balanced_prior_loss(logits, target, kind)
                loss.backward()
                self.assertLess(logits.grad[0, 0, 4, 4].item(), 0)
                self.assertTrue((logits.grad[1] > 0).all())
                self.assertTrue(torch.isfinite(loss))

    def test_pair_pixel_and_geometry_chunk_boundaries_preserve_targets(self):
            boxes = torch.tensor([[.2, 1.3, 10.8, 12.2], [3.1, 4.2, 15.3, 12.9],
                                  [4.5, 4.1, 7.8, 7.7], [-3., -1., 2.5, 3.2], [8., 7., 5., 5.]])
            for sigma in (0, 1.3):
                expected = build_overlap_prior(boxes, (13, 19), sigma)
                actual = build_overlap_prior(boxes, (13, 19), sigma,
                                             pair_chunk_size=1, pixel_chunk_size=7, geometry_chunk_size=2)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gaussian_preserves_cudnn_flags_and_restores_on_error(self):
            cudnn = torch.backends.cudnn
            keys = ('enabled', 'benchmark', 'benchmark_limit', 'deterministic', 'allow_tf32')

            def state():
                return {key: getattr(cudnn, key) for key in keys}

            boxes = torch.tensor([[1.,1.,10.,10.],[3.,3.,12.,12.]])
            original_conv = torch.nn.functional.conv2d
            for enabled, benchmark, limit, deterministic in ((True,True,7,True), (False,False,3,False)):
                with self.subTest(enabled=enabled), cudnn.flags(
                    enabled=enabled, benchmark=benchmark, benchmark_limit=limit,
                    deterministic=deterministic, allow_tf32=True,
                ):
                    before = state()
                    calls = []

                    def observe(*args, **kwargs):
                        calls.append(state())
                        self.assertEqual(state(), {**before, 'allow_tf32':False})
                        return original_conv(*args, **kwargs)

                    with patch('ultralytics.utils.overlap_prior.F.conv2d', side_effect=observe):
                        build_overlap_prior(boxes,(16,16),sigma=1.3)
                    self.assertEqual(len(calls),2)
                    self.assertEqual(state(),before)

                    def fail(*args, **kwargs):
                        self.assertEqual(state(), {**before, 'allow_tf32':False})
                        raise RuntimeError('convolution fixture failure')

                    with patch('ultralytics.utils.overlap_prior.F.conv2d',side_effect=fail):
                        with self.assertRaisesRegex(RuntimeError,'convolution fixture failure'):
                            build_overlap_prior(boxes,(16,16),sigma=1.3)
                    self.assertEqual(state(),before)

    def test_ragged_batch_shuffled_labels_empty_and_singleton_images(self):
            batch = make_batch()
            batch['img'] = torch.randn(4, 3, 64, 96)
            batch['bboxes'] = torch.tensor([[.4,.5,.4,.4], [.5,.5,.4,.4], [.6,.4,.5,.5],
                                           [.3,.3,.2,.2], [.5,.5,.6,.6], [.4,.5,.4,.4], [.7,.7,.3,.3]])
            batch['batch_idx'] = torch.tensor([0,0,0,2,3,3,3])
            indices = batch['batch_idx']
            boxes = batch['bboxes']
            xyxy = torch.cat((boxes[:,:2]-boxes[:,2:]/2, boxes[:,:2]+boxes[:,2:]/2),1)
            xyxy *= torch.tensor([96,64,96,64])
            expected = torch.stack([build_overlap_prior(xyxy[indices==i], (64,96), 1.3) for i in range(4)])[:,None]
            expected = torch.nn.functional.interpolate(expected, (8,12), mode='area')
            order = torch.tensor([5,0,3,6,1,4,2])
            batch['bboxes'] = batch['bboxes'][order]
            batch['batch_idx'] = batch['batch_idx'][order]
            actual = batch_overlap_targets(batch, (8,12), 'cpu', 1.3,
                                           pair_chunk_size=2, pixel_chunk_size=113, geometry_chunk_size=3)
            # Batched Gaussian convolution can change FP32 accumulation order.
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(actual[1:3].count_nonzero().item(), 0)
            batch['bboxes'] = torch.empty(0,4)
            batch['batch_idx'] = torch.empty(0,dtype=torch.long)
            self.assertEqual(batch_overlap_targets(batch,(8,12),'cpu').count_nonzero().item(),0)

    def test_target_no_grad_under_amp_preserves_prior_head_gradients(self):
            devices = ['cpu']
            for device in devices:
                with self.subTest(device=device):
                    batch = make_batch()
                    batch['bboxes'] = batch['bboxes'].to(device).requires_grad_()
                    batch['batch_idx'] = batch['batch_idx'].to(device)
                    head = OverlapPriorHead(4).to(device)
                    feature = torch.randn(2,4,8,12,device=device,requires_grad=True)
                    dtype = torch.float16 if device=='cuda' else torch.bfloat16
                    with torch.autocast(device_type=device,dtype=dtype):
                        logits = head(feature)
                        target = batch_overlap_targets(batch,(8,12),device)
                        loss = balanced_prior_loss(logits,target)
                    self.assertFalse(target.requires_grad)
                    self.assertEqual(target.dtype,torch.float32)
                    self.assertEqual(target.device.type,device)
                    loss.backward()
                    self.assertIsNone(batch['bboxes'].grad)
                    self.assertGreater(head.layers[-1].weight.grad.abs().sum().item(),0)
                    self.assertGreater(feature.grad.abs().sum().item(),0)

    def test_channel_split_preserves_identity_half_and_conditions_only_fused_half(self):
            for channels in (16, 17):
                with self.subTest(channels=channels):
                    fusion = ChannelSplitPriorFusion(channels).eval()
                    feature = torch.randn(2, channels, 8, 12, requires_grad=True)
                    prior = torch.rand(2, 1, 8, 12, requires_grad=True)
                    output = fusion(feature, prior)
                    split = channels // 2
                    self.assertEqual(output.shape, feature.shape)
                    torch.testing.assert_close(output[:, :split], feature[:, :split], atol=0, rtol=0)
                    changed = fusion(feature, 1 - prior)
                    torch.testing.assert_close(output[:, :split], changed[:, :split], atol=0, rtol=0)
                    self.assertFalse(torch.equal(output[:, split:], changed[:, split:]))
                    output[:, split:].square().mean().backward()
                    # The processed branch cannot alter or read the identity half.
                    self.assertEqual(feature.grad[:, :split].count_nonzero().item(), 0)
                    self.assertGreater(feature.grad[:, split:].abs().sum().item(), 0)
                    self.assertGreater(prior.grad.abs().sum().item(), 0)

    def test_new_prior_fusion_reaches_the_prior_head_without_auxiliary_loss(self):
            original = FeatureEnhancementBase(16, patch_size=(1, 2, 3), gamma_init=.02).eval()
            conditioned = MSFEM(16).eval()
            incompatible = conditioned.load_state_dict(original.state_dict(), strict=False)
            self.assertFalse(incompatible.unexpected_keys)
            self.assertTrue(incompatible.missing_keys)
            self.assertTrue(all(k.startswith('prior_fusion.') for k in incompatible.missing_keys))
            feature = torch.randn(2, 16, 8, 12)
            with torch.no_grad():
                low = conditioned([feature, torch.full((2, 1, 8, 12), -10.0)])
                high = conditioned([feature, torch.full((2, 1, 8, 12), 10.0)])
                self.assertFalse(torch.equal(low, high))
            head = OverlapPriorHead(16)
            conditioned([feature, head(feature)]).square().mean().backward()
            self.assertGreater(head.layers[-1].weight.grad.abs().sum().item(), 0)

    def test_full_models_backward_validation_checkpoint_and_export_mode(self):
            for config in ("ormsfd.yaml",):
                model = DetectionModel(str(ROOT / config), verbose=False)
                model.args = copy.deepcopy(DEFAULT_CFG)
                batch = make_batch()
                prediction = model(batch["img"])
                loss, items = model.loss(batch, prediction)
                self.assertEqual(len(items), 4)
                self.assertTrue(torch.isfinite(loss).all())
                loss.sum().backward()
                prior_head = next(m for m in model.model if isinstance(m, OverlapPriorHead))
                self.assertGreater(prior_head.layers[-1].weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(isinstance(m.prior_fusion, ChannelSplitPriorFusion)
                                    for m in model.model if isinstance(m, MSFEM)))
                model.eval()
                with torch.no_grad():
                    result = model(batch["img"])
                    self.assertEqual(len(model.loss(batch, result)[1]), 4)
                    self.assertEqual(result[1]["overlap_prior"].shape, (2, 1, 8, 12))
                    # No graph tensors cached on modules: deepcopy and checkpoint work.
                    clone = copy.deepcopy(model)
                    torch.testing.assert_close(clone(batch["img"])[0], result[0])
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "model.pt"
                        torch.save(model, path)
                        loaded = torch.load(path, weights_only=False)
                        torch.testing.assert_close(loaded(batch["img"])[0], result[0])
                    model.model[-1].export = True
                    exported = model(batch["img"])
                    self.assertIsInstance(exported, torch.Tensor)
                    torch.testing.assert_close(exported, result[0])

    def test_no_prior_supervision_skips_targets_and_keeps_detection_gradients(self):
            import os
            model = DetectionModel(str(ROOT / 'ormsfd.yaml'), verbose=False)
            model.yaml['overlap_prior']['weight'] = 0.0
            model.args = copy.deepcopy(DEFAULT_CFG)
            batch = make_batch()
            with patch.dict(os.environ, {'BBOX_LOSS_TYPE':'mpdiou','USE_REPLOSS':'1',
                                         'REP_ALPHA':'0.0001','REP_BETA':'0'}), \
                 patch('ultralytics.utils.loss.batch_overlap_targets', side_effect=AssertionError('GT map must be skipped')), \
                 patch('ultralytics.utils.loss.balanced_prior_loss', side_effect=AssertionError('BCE must be skipped')):
                prediction = model(batch['img'])
                loss, items = model.loss(batch, prediction)
                self.assertTrue(torch.isfinite(loss).all())
                self.assertEqual(items.numel(), 4)
                self.assertEqual(items[-1].item(), 0.)
                self.assertEqual(loss[-1].item(), 0.)
                self.assertEqual(model.criterion.bbox_loss.bbox_loss_type, 'mpdiou')
                self.assertTrue(model.criterion.use_repulsion_loss)
                self.assertEqual(model.criterion.rep_alpha, 1e-4)
                loss.sum().backward()
                prior_head = next(m for m in model.model if isinstance(m, OverlapPriorHead))
                self.assertGreater(prior_head.layers[-1].weight.grad.abs().sum().item(), 0)
                model.eval()
                with torch.no_grad():
                    self.assertEqual(model.loss(batch, model(batch['img']))[1][-1].item(), 0.)


if __name__ == "__main__":
    unittest.main()
