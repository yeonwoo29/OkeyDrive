# OkeyDrive Runbook

## Supported environments

The upstream SparseDrive instructions use Linux, Python 3.8, PyTorch 1.13.0 with CUDA 11.6, and a compiled deformable aggregation operator. Use that combination for the integrated nuScenes path. The current Windows editing environment can run independent PyTorch tests, but its Python 3.13 environment cannot install the pinned legacy MMCV stack.

Use WSL2 or Linux for full training and evaluation:

```bash
conda create -n okeydrive python=3.8 -y
conda activate okeydrive
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements-okeydrive.txt
cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../..
```

OkeyDrive uses the Mamba-1 `selective_scan_fn` interface. It does not instantiate Mamba-2 or Mamba-3. If the pinned package does not build a CUDA kernel for the exact PyTorch/CUDA ABI, build it in the same environment:

```bash
MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm==1.2.0.post1 --no-build-isolation
python -c "from okeydrive.mamba import cuda_selective_scan_available; assert cuda_selective_scan_available()"
```

Do not silently use `scan_backend="reference"` for performance experiments. That backend is an equation-level CPU/GPU test implementation.

## Data and checkpoints

Keep private roots outside the repository. Create local links without recording their resolved targets:

```bash
mkdir -p data checkpoints outputs
ln -s "$DATA_ROOT" data/nuscenes
ln -s "$CHECKPOINT_ROOT/diffusiondrive_nusc_stage2.pth" checkpoints/diffusiondrive_nusc_stage2.pth
```

Generate nuScenes metadata and upstream anchors from the training split as described in `scripts/create_data.sh` and `scripts/kmeans.sh`. Do not generate anchors from validation data.

## A. Baseline smoke test and evaluation

```bash
python tools/test.py projects/configs/okeydrive/ablations/01_original_diffusiondrive.py checkpoints/diffusiondrive_nusc_stage2.pth --eval bbox --metrics-out outputs/baseline_metrics.json
```

## B. External pretraining

CarFusion uses the verified 12-point COCO-style conversion; COCO uses native person keypoints. Either dataset pair may be provided, but the full method uses both:

```bash
python tools/okeydrive/pretrain_external.py \
  --carfusion-annotations "$CARFUSION_TRAIN_JSON" \
  --carfusion-images "$CARFUSION_IMAGE_ROOT" \
  --coco-annotations "$COCO_TRAIN_JSON" \
  --coco-images "$COCO_IMAGE_ROOT" \
  --output-dir outputs/pretrain \
  --deterministic --seed 0
```

The training checkpoint contains optimizer state for resume and is excluded from release export. Exported checkpoints are reduced to state dict plus a non-identifying hash.

The three external objectives default to equal weights. Override them explicitly with `--visibility-weight`, `--localization-weight`, and `--reconstruction-weight`; the selected values are stored with the training metrics.

## C. Initial OkeyDrive training

```bash
python tools/train.py projects/configs/okeydrive/okeydrive_initial_stage2.py --deterministic --seed 0 --work-dir outputs/initial
ln -s ../outputs/initial/latest.pth checkpoints/okeydrive_initial_stage2.pth
```

Expected missing checkpoint keys must be limited to `head.okeydrive.*` and `head.motion_plan_head.okeydrive_refiner.*`. Use `tools/okeydrive/checkpoint_compat.py` before training; unrelated missing or unexpected keys are fatal in that tool.

```bash
python tools/okeydrive/checkpoint_compat.py \
  projects/configs/okeydrive/okeydrive_small_stage2.py \
  checkpoints/diffusiondrive_nusc_stage2.pth \
  --output outputs/checkpoint_compat.json
```

## D. Joint fine-tuning

```bash
python tools/train.py projects/configs/okeydrive/okeydrive_joint_stage2.py --deterministic --seed 0 --work-dir outputs/joint
ln -s ../outputs/joint/latest.pth checkpoints/okeydrive_joint_stage2.pth
```

The upstream runner provides optimizer, cosine scheduler, AMP, checkpoint/resume, train/eval modes, and temporal scene handling. Use `--resume-from` only with a matching joint-training checkpoint.

## E. Official full validation

```bash
python tools/test.py projects/configs/okeydrive/okeydrive_small_stage2.py checkpoints/okeydrive_joint_stage2.pth --eval bbox --out outputs/full_val_predictions.pkl --metrics-out outputs/full_val_metrics.json --deterministic --seed 0
```

Run all ablations with the same split, checkpoint policy, input resolution, evaluator, valid-sample mask, and seed:

```bash
python tools/okeydrive/run_ablations.py --checkpoint-root checkpoints --output-root outputs/ablations
```

Seed-specific checkpoints use `<condition>_seed<seed>.pth`; a shared checkpoint without the seed suffix is accepted when only evaluation randomness is being repeated. The runner records the config label, seed, checkpoint hash, raw metrics, and measured mean/population-standard-deviation summaries. It never creates a mean or standard deviation until the requested runs finish successfully.

Launch isolated repeated training runs as follows:

```bash
python tools/okeydrive/run_repeated_training.py \
  projects/configs/okeydrive/okeydrive_joint_stage2.py \
  --output-root outputs/repeated --seeds 0 1 2
```

Additional controlled sensitivity configs are under `projects/configs/okeydrive/sensitivity`: frozen CLIP, `eta=0.35`, `eta=0.65`, and reversed canonical candidate sequence. These are not part of the seven required ablations and need separately trained matching checkpoints.

Mini/smoke, official full validation, and any occlusion subset must use separate result files. Keep research targets separate from evaluator-produced measurements.

## Tests, latency, and release

```bash
python -m unittest discover -s tests -v
python tools/okeydrive/benchmark_e2e.py projects/configs/okeydrive/okeydrive_small_stage2.py checkpoints/okeydrive_joint_stage2.pth --output outputs/latency.json
python tools/okeydrive/privacy.py .
python tools/okeydrive/export_release.py --source . --output dist/okeydrive.zip
python tools/okeydrive/privacy.py dist/okeydrive.zip
```

The benchmark includes CLIP, keypoints, BEV lifting/fusion, candidate aggregation, and Mamba. It records warm-up count, synchronization, batch size, input resolution, and measured end-to-end latency. It does not label reference-scan timing as CUDA-kernel timing.
