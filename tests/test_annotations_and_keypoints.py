from __future__ import annotations

import unittest

import torch

from okeydrive.annotations import (
    CARFUSION_CANONICAL_SOURCE_INDICES,
    COCO_PERSON_SOURCE_INDICES,
    build_part_visibility_targets,
    canonicalize_carfusion,
    canonicalize_coco_person,
)
from okeydrive.keypoints import masked_smooth_l1_loss, recover_keypoints
from okeydrive.visibility import remap_semantic_parts_for_flip


class AnnotationAndKeypointTests(unittest.TestCase):
    def test_verified_source_index_mappings(self):
        vehicle = torch.stack(
            [torch.tensor([float(index), -float(index), float(index % 3)]) for index in range(12)]
        )
        vehicle_xy, vehicle_state = canonicalize_carfusion(vehicle)
        self.assertEqual(vehicle_xy[:, 0].tolist(), list(CARFUSION_CANONICAL_SOURCE_INDICES))
        self.assertEqual(
            vehicle_state.tolist(),
            [index % 3 for index in CARFUSION_CANONICAL_SOURCE_INDICES],
        )

        person = torch.stack(
            [torch.tensor([float(index), 0.0, 2.0]) for index in range(17)]
        )
        person_xy, _ = canonicalize_coco_person(person)
        self.assertEqual(person_xy[:, 0].tolist(), list(COCO_PERSON_SOURCE_INDICES))

    def test_visibility_state_zero_is_unsupervised_and_one_is_occluded(self):
        states = torch.tensor([0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        target, mask = build_part_visibility_targets(states)
        self.assertTrue(torch.allclose(target, torch.tensor([0.5, 0.0, 0.0, 1.0])))
        self.assertEqual(mask.tolist(), [True, False, True, True])

    def test_hard_gate_preserves_visible_points_exactly(self):
        initial = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        reconstructed = torch.full_like(initial, -7.0)
        visibility = torch.tensor([0.7, 0.2, 0.9, 0.1])
        recovered, reliable = recover_keypoints(initial, reconstructed, visibility, threshold=0.5)
        self.assertTrue(torch.equal(recovered[:3], initial[:3]))
        self.assertTrue(torch.equal(recovered[6:9], initial[6:9]))
        self.assertTrue(torch.equal(recovered[3:6], reconstructed[3:6]))
        self.assertEqual(reliable.squeeze(-1).tolist(), [True] * 3 + [False] * 3 + [True] * 3 + [False] * 3)

    def test_all_visible_all_occluded_and_exact_threshold(self):
        initial = torch.rand(12, 2)
        reconstructed = torch.rand(12, 2)
        all_visible, _ = recover_keypoints(initial, reconstructed, torch.ones(4))
        all_occluded, _ = recover_keypoints(initial, reconstructed, torch.zeros(4))
        boundary, reliable = recover_keypoints(
            initial, reconstructed, torch.full((4,), 0.5), threshold=0.5
        )
        self.assertTrue(torch.equal(all_visible, initial))
        self.assertTrue(torch.equal(all_occluded, reconstructed))
        self.assertTrue(torch.equal(boundary, initial))
        self.assertTrue(reliable.all())

    def test_masked_loss_ignores_unlabeled_values(self):
        prediction = torch.zeros(2, 2)
        target_a = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
        target_b = torch.tensor([[1.0, 1.0], [-100.0, -100.0]])
        mask = torch.tensor([True, False])
        self.assertEqual(
            masked_smooth_l1_loss(prediction, target_a, mask),
            masked_smooth_l1_loss(prediction, target_b, mask),
        )

    def test_horizontal_flip_swaps_semantic_left_and_right(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(
            remap_semantic_parts_for_flip(values, True).tolist(),
            [[2.0, 1.0, 4.0, 3.0]],
        )
        self.assertTrue(torch.equal(remap_semantic_parts_for_flip(values, False), values))


if __name__ == "__main__":
    unittest.main()
