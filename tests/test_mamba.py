from __future__ import annotations

import unittest

import torch

from okeydrive.mamba import (
    CandidateVisibilityAggregator,
    ContextMotionMambaRefiner,
    VisibilityConditionedMambaBlock,
    apply_candidate_permutation,
    canonical_candidate_order,
    cuda_selective_scan_available,
    selective_scan_cuda,
    selective_scan_reference,
)


class MambaTests(unittest.TestCase):
    def test_reference_scan_matches_manual_scalar_recurrence(self):
        inputs = torch.tensor([[[2.0], [3.0]]])
        delta = torch.tensor([[[0.5], [0.25]]])
        A = torch.tensor([[-1.0]])
        B = torch.ones(1, 2, 1)
        C = torch.ones(1, 2, 1)
        D = torch.zeros(1)
        result = selective_scan_reference(inputs, delta, A, B, C, D)
        state0 = 0.5 * 2.0
        state1 = torch.exp(torch.tensor(-0.25)) * state0 + 0.25 * 3.0
        self.assertTrue(torch.allclose(result[0, :, 0], torch.tensor([state0, state1])))

    def test_delta_is_positive_modulation_is_bounded_and_no_evidence_is_neutral(self):
        torch.manual_seed(4)
        block = VisibilityConditionedMambaBlock(
            model_dim=8, state_dim=4, expand=1, visibility_dim=10, scan_backend="reference"
        )
        tokens = torch.randn(2, 5, 8)
        visibility = torch.rand(2, 5, 10)
        _, diagnostics = block(
            tokens, visibility, torch.zeros(2, 5, 1, dtype=torch.bool), return_diagnostics=True
        )
        self.assertTrue((diagnostics.delta > 0).all())
        self.assertEqual(float(diagnostics.delta_modulation.abs().max()), 0.0)
        self.assertEqual(float(diagnostics.B_modulation.abs().max()), 0.0)
        self.assertEqual(float(diagnostics.C_modulation.abs().max()), 0.0)
        _, visible_diagnostics = block(
            tokens, visibility, torch.ones(2, 5, 1, dtype=torch.bool), return_diagnostics=True
        )
        self.assertLessEqual(float(visible_diagnostics.delta_modulation.abs().max()), 0.5)

    def test_visibility_changes_all_conditioned_parameters_and_output(self):
        torch.manual_seed(7)
        block = VisibilityConditionedMambaBlock(
            model_dim=8, state_dim=4, expand=1, visibility_dim=10, scan_backend="reference"
        )
        tokens = torch.randn(1, 5, 8)
        low = torch.zeros(1, 5, 10)
        high = torch.ones(1, 5, 10)
        low_output, low_diag = block(tokens, low, torch.ones(1, 5, 1), True)
        high_output, high_diag = block(tokens, high, torch.ones(1, 5, 1), True)
        self.assertFalse(torch.allclose(low_diag.delta, high_diag.delta))
        self.assertFalse(torch.allclose(low_diag.B, high_diag.B))
        self.assertFalse(torch.allclose(low_diag.C, high_diag.C))
        self.assertFalse(torch.equal(low_output, high_output))

    def test_no_modulation_is_exactly_visibility_independent(self):
        torch.manual_seed(8)
        block = VisibilityConditionedMambaBlock(
            model_dim=8,
            state_dim=4,
            expand=1,
            visibility_dim=10,
            scan_backend="reference",
            use_modulation=False,
        )
        tokens = torch.randn(1, 5, 8)
        first = block(tokens, torch.zeros(1, 5, 10), torch.ones(1, 5, 1))
        second = block(tokens, torch.ones(1, 5, 10), torch.ones(1, 5, 1))
        self.assertTrue(torch.equal(first, second))

    def test_empty_object_list_produces_explicit_no_evidence(self):
        aggregator = CandidateVisibilityAggregator(x_bound=(-2.0, 2.0, 1.0), y_bound=(-2.0, 2.0, 1.0))
        trajectories = torch.zeros(2, 3, 6, 2)
        visibility, evidence, iou = aggregator(
            trajectories,
            torch.zeros(2, 0, 5),
            torch.zeros(2, 0, 4),
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(2, 0, dtype=torch.bool),
        )
        self.assertEqual(visibility.shape, (2, 3, 8))
        self.assertFalse(evidence.any())
        self.assertEqual(iou.shape, (2, 3, 6, 0))

    @unittest.skipUnless(cuda_selective_scan_available(), "Mamba-1 CUDA selective scan is unavailable")
    def test_cuda_scan_matches_reference_forward_and_backward(self):
        torch.manual_seed(9)
        inputs = torch.randn(2, 5, 4, device="cuda", requires_grad=True)
        delta = torch.rand(2, 5, 4, device="cuda", requires_grad=True) + 0.01
        A = -torch.rand(4, 3, device="cuda")
        B = torch.randn(2, 5, 3, device="cuda", requires_grad=True)
        C = torch.randn(2, 5, 3, device="cuda", requires_grad=True)
        D = torch.randn(4, device="cuda", requires_grad=True)
        reference = selective_scan_reference(inputs, delta, A, B, C, D)
        cuda = selective_scan_cuda(inputs, delta, A, B, C, D)
        self.assertTrue(torch.allclose(reference, cuda, atol=2e-4, rtol=2e-4))
        reference.sum().backward(retain_graph=True)
        reference_gradient = inputs.grad.detach().clone()
        inputs.grad.zero_()
        cuda.sum().backward()
        self.assertTrue(torch.allclose(reference_gradient, inputs.grad, atol=5e-4, rtol=5e-4))

    def test_canonical_order_is_stable_and_invertible(self):
        trajectories = torch.zeros(1, 3, 2, 2)
        trajectories[0, :, -1, 0] = torch.tensor([-1.0, 1.0, 0.0])
        permutation, inverse = canonical_candidate_order(trajectories, torch.tensor([2, 0, 1]))
        self.assertEqual(permutation.tolist(), [[0, 2, 1]])
        restored = torch.gather(permutation, 1, inverse)
        self.assertEqual(restored.tolist(), [[0, 1, 2]])

    def test_canonical_sort_is_invariant_to_input_storage_permutation(self):
        trajectories = torch.zeros(1, 4, 2, 2)
        trajectories[0, :, -1, 0] = torch.tensor([1.0, -2.0, 0.5, -1.0])
        stable_ids = torch.tensor([10, 11, 12, 13])
        first_order, _ = canonical_candidate_order(trajectories, stable_ids)
        first = apply_candidate_permutation(trajectories, first_order)
        storage = torch.tensor([[2, 0, 3, 1]])
        shuffled_trajectories = apply_candidate_permutation(trajectories, storage)
        shuffled_ids = apply_candidate_permutation(stable_ids[None], storage)
        second_order, _ = canonical_candidate_order(shuffled_trajectories, shuffled_ids)
        second = apply_candidate_permutation(shuffled_trajectories, second_order)
        self.assertTrue(torch.equal(first, second))

    def test_all_anchor_modes_expand_to_ten_candidates_for_all_commands(self):
        torch.manual_seed(1)
        refiner = ContextMotionMambaRefiner(
            model_dim=8,
            future_steps=6,
            base_modes=6,
            num_candidates=10,
            state_dim=4,
            scan_backend="reference",
            x_bound=(-4.0, 4.0, 1.0),
            y_bound=(-2.0, 8.0, 1.0),
        )
        base = torch.randn(1, 3, 6, 6, 2).cumsum(dim=-2) * 0.1
        navigation = torch.randn(1, 3, 6, 8)
        sparse = torch.randn(1, 8)
        objects = torch.tensor([[[0.0, 1.0, 2.0, 4.0, 0.0]]])
        output = refiner(
            base,
            navigation,
            sparse,
            None,
            objects,
            torch.rand(1, 1, 4),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
        )
        self.assertEqual(output["prediction"].shape, (1, 3, 10, 6, 2))
        self.assertEqual(output["score"].shape, (1, 3, 10))
        self.assertEqual(output["base_increment"].shape, (1, 3, 10, 6, 2))
        self.assertTrue(torch.isfinite(output["prediction"]).all())


if __name__ == "__main__":
    unittest.main()
