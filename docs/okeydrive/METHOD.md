# OkeyDrive Method and Integration

## Upstream path

This implementation is based on DiffusionDrive `nusc` commit `ae54fd87b32b3762f20e63ffd0af91d343cade85`. The upstream ResNet-50/FPN, sparse detection head, sparse map head, temporal instance queue, motion head, navigation command, decoder, and official planning evaluator are retained. The upstream depth branch is auxiliary supervision only, so OkeyDrive explicitly uses its predicted depth in a calibrated lifting module.

The initial detector runs once on the upstream image features. Its current-frame predictions provide proposals; no enhanced BEV is required to obtain those proposals. OkeyDrive then enhances the first FPN level, constructs BEV, updates sparse object features, and runs one shared-weight detection refinement. The map and planning heads consume the enhanced features. The temporal detector cache is refreshed with the enhanced sparse features and final refined predictions, while retaining the upstream cache structure and tensor contract.

## Runtime data flow

1. Normalized multi-view RGB `images [B,V,3,H,W]` enters the upstream backbone and FPN.
2. The upstream detector produces encoded 3D proposals and class scores.
3. Top supported predicted vehicle/pedestrian proposals are projected with augmented `lidar2img` matrices to aligned RGB and FPN ROIs.
4. The actual pretrained CLIP ViT-B/16 image and text encoders process RGB crops and class-specific part prompts.
5. Independent learned sigmoid calibration maps cosine scores to `visibility [B,V,O,4]`.
6. A class-conditioned localizer predicts crop-normalized `keypoints [B,V,O,12,2]`; a class-conditioned autoencoder reconstructs them.
7. A hard mask at `eta=0.5` gives `K_recovered = where(M, K_initial, K_reconstructed)`. Recovered geometry never overwrites observed visibility evidence.
8. Foreground ROI tokens query geometric key/value tokens containing coordinates, semantic part, class, visibility, and recovery state. Normalized ROI residuals are scattered only inside valid boxes.
9. Predicted depth and inverse augmented projection matrices unproject feature pixels into the lidar/ego frame and splat them into `BEV [B,C,H_bev,W_bev]`.
10. BEV samples refine sparse detector tokens and provide candidate-wise planning context.
11. Ten command-conditioned trajectory candidates are produced by a learned interpolation over all six upstream anchors. The implementation does not truncate an anchor prefix and does not cluster validation data.
12. Candidates are sorted left-to-right, processed by visibility-modulated Mamba-1, restored to stable anchor order, scored, and decoded. The OkeyDrive branch does not construct diffusion noise, timesteps, or a denoising loop.

## Part and keypoint schemas

Vehicle parts are front-left, front-right, rear-left, and rear-right. The inspected CarFusion conversion exposes 12 retained landmarks in this source order: rear wheels, front wheels, rear lights, front lights, rear roof points, and front roof points, each left then right. Canonical OkeyDrive order groups wheel/light/roof for each semantic corner. Raw identifiers omitted by the inspected conversion are not synthesized.

Pedestrian parts are left arm, right arm, left leg, and right leg. Canonical triples use COCO indices `(5,7,9)`, `(6,8,10)`, `(11,13,15)`, and `(12,14,16)` in zero-based indexing. Face landmarks are excluded. COCO state 0 is unlabeled, state 1 is labeled but occluded, and state 2 is visible. A part target is the visible fraction among labeled points; a part with no labeled points is masked from the loss.

A horizontal flip swaps semantic left/right slots with permutation `(1,0,3,2)`. External pretraining transforms x coordinates and keypoint slots together, records the flip flag, and applies the same part permutation to CLIP outputs. This is semantic object left/right, not display left/right.

## Coordinates and geometry

Upstream planning uses `(x,y)` with +y forward, confirmed by the decoder's initial yaw of `pi/2`. Ego left is therefore `(-1,0)`. Signed lateral coordinates are the final waypoint dot this vector; positive-left values sort descending. Exact ties use stable generated anchor IDs, never input array position.

Image ROIs are in augmented pixel coordinates. Keypoints are normalized inside each ROI and converted back with the same box when needed. BEV lifting uses predicted camera depth and the inverse augmented `lidar2img`; normalized 2D coordinates are never treated as BEV coordinates. The default grid covers x `[-15,15)` and y `[-10,50)` at 0.5 m. Candidate/object association uses rasterized oriented footprint IoU in this BEV frame, not camera-box IoU.

## Visibility aggregation and selective SSM

Multi-view probabilities are averaged only over valid projections of the same predicted 3D proposal. Vehicle and pedestrian channels remain separate. For each candidate, oriented ego footprints at six waypoints are compared with predicted object BEV boxes. IoU is weighted from early to late waypoints and normalized over time and objects. If no predicted object overlaps, the evidence mask is false and modulation is exactly neutral.

For motion tokens `x [B,Z,D]`, Mamba produces positive `Delta [B,Z,D_inner]` after its bias and a single softplus, plus input-dependent `B,C [B,Z,D_state]`. The conditioner computes

`s_m = 0.5 * tanh(psi_m([P(x), V(p,x)]))`, for `m in {Delta,B,C}`,

then applies `Delta_hat=(1+s_Delta)Delta`, `B_hat=(1+s_B)B`, and `C_hat=(1+s_C)C`. Stable `A=-exp(A_log)` is unchanged. The CUDA call receives the already-positive Delta with `delta_bias=None` and `delta_softplus=False`, preventing duplicate softplus. The Python reference scan uses the identical recurrence for tests.

Sequence index `z` is candidate order, not time. State `h[z-1]` belongs to the previous candidate. No temporal claim is made for this scan.

## Training losses and inference selection

Upstream detection, map, motion, planning-status, and depth losses remain. External pretraining adds masked part BCE, masked keypoint Smooth L1, and masked reconstruction Smooth L1. NuScenes has no native fine-grained part/keypoint supervision, so those GT losses are not fabricated there. Candidate assignment uses the training future trajectory through the upstream sampler. Inference uses the learned candidate score plus the existing route command and collision-aware predicted-motion rescore; it never chooses the GT-nearest candidate.

## Evaluator definition

The retained evaluator computes per-waypoint Euclidean L2 and collision indicators at 0.5 s intervals. It converts each six-value series to cumulative means, reports the cumulative values at 1 s, 2 s, and 3 s, and averages those three reported values. Collision values are ratios internally and formatted as percentages only for display.

## Feature flags and ablations

The seven configs under `projects/configs/okeydrive/ablations` isolate original diffusion, recovered-keypoint BEV with diffusion, Mamba only, initial keypoints, AE-only keypoints, recovered keypoints without modulation, and full OkeyDrive. Turning every OkeyDrive feature off leaves the upstream feature objects unchanged. Separate sensitivity configs cover frozen CLIP, two alternate hard-gate thresholds, and reversed canonical candidate order. Accuracy improvements are not assumed; they require full validation measurements.
