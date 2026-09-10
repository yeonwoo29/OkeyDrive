from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from okeydrive.checkpoint import categorize_checkpoint_keys, state_dict_sha256
from okeydrive.runtime_privacy import redact_local_paths
from tools.okeydrive.export_release import export_checkpoint, write_reproducible_zip
from tools.okeydrive.privacy import scan_archive, scan_tree, summarized_report


REPOSITORY = Path(__file__).resolve().parents[1]


class ReleaseAndIntegrationTests(unittest.TestCase):
    def test_checkpoint_categorization_and_dtype_independent_hashing(self):
        result = categorize_checkpoint_keys(
            ["head.okeydrive.layer.weight", "head.det_head.bad"], ["obsolete.weight"]
        )
        self.assertEqual(result["expected_new_module_missing"], ["head.okeydrive.layer.weight"])
        self.assertEqual(result["incompatible_missing"], ["head.det_head.bad"])
        self.assertEqual(result["unexpected"], ["obsolete.weight"])
        digest = state_dict_sha256({"scalar": torch.tensor(1.0, dtype=torch.bfloat16)})
        self.assertEqual(len(digest), 64)

    def test_runtime_redaction_removes_absolute_user_roots(self):
        windows_path = "C:" + "\\Users\\" + "private_name\\project\\file.py"
        posix_path = "/" + "home/private_name/project/file.py"
        sanitized = redact_local_paths(f"first={windows_path} second={posix_path}")
        self.assertNotIn("private_name", sanitized)
        self.assertEqual(sanitized.count("<LOCAL_PATH>"), 2)

    def test_release_checkpoint_strips_optimizer_and_archive_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_checkpoint = root / "training.pt"
            release_checkpoint = root / "state.pt"
            torch.save(
                {"model": {"weight": torch.ones(2)}, "optimizer": {"private": "discard"}},
                source_checkpoint,
            )
            export_checkpoint(source_checkpoint, release_checkpoint)
            payload = torch.load(release_checkpoint, map_location="cpu")
            self.assertEqual(set(payload), {"state_dict", "metadata"})
            source = root / "source"
            source.mkdir()
            (source / "README.md").write_text("Anonymous research release.\n", encoding="utf-8")
            archive = root / "release.zip"
            write_reproducible_zip(source, archive)
            self.assertEqual(scan_archive(archive), [])
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(handle.namelist(), ["README.md"])

    def test_privacy_report_never_repeats_sensitive_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sensitive = "C:" + "\\Users\\" + "private_name\\secret.txt"
            (root / "note.txt").write_text(sensitive, encoding="utf-8")
            findings = scan_tree(root)
            report = str(summarized_report(findings))
            self.assertTrue(findings)
            self.assertNotIn("private_name", report)

    def test_integrated_path_returns_before_diffusion_noise(self):
        source = (REPOSITORY / "projects/mmdet3d_plugin/models/motion/motion_planning_head_v13.py").read_text(
            encoding="utf-8"
        )
        train_start = source.index("    def forward_train(")
        test_start = source.index("    def forward_test(")
        loss_start = source.index("    def loss(", test_start)
        train_section = source[train_start:test_start]
        test_section = source[test_start:loss_start]
        for section in (train_section, test_section):
            branch = section.index("if self.okeydrive_refiner is not None:")
            early_return = section.index("return motion_output, planning_output", branch)
            diffusion_noise = section.index("self.diffusion_scheduler.add_noise", early_return)
            self.assertLess(early_return, diffusion_noise)

    def test_all_required_ablation_configs_exist(self):
        paths = sorted((REPOSITORY / "projects/configs/okeydrive/ablations").glob("*.py"))
        self.assertEqual(len(paths), 7)
        names = [path.stem for path in paths]
        self.assertEqual(names[0], "01_original_diffusiondrive")
        self.assertEqual(names[-1], "07_full_okeydrive")


if __name__ == "__main__":
    unittest.main()
