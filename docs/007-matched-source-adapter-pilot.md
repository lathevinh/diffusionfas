# 007 - Matched source-only adapter pilot

**Pilot hoàn tất, không có PAD accuracy result.**
[Aggregate evidence](../results/adapter-pilot-v1.json) ghi `completed: true`;
checkpoints và individual validation scores giữ private.

## Kết quả đo được

Source validation epsilon-MSE của best EMA; cả hai best checkpoints đều thuộc
epoch 2, sau đúng 360 optimizer steps:

| Model | CASIA (12) | MSU (6) | SiW (99) | Domain-macro |
|---|---:|---:|---:|---:|
| Frozen backbone | 0,008992091 | 0,010425643 | 0,012033171 | 0,010483635 |
| Spatial adapter | 0,008984428 | 0,010415282 | 0,012022671 | 0,010474127 |
| Temporal adapter | 0,008965238 | 0,010394373 | 0,012006165 | 0,010455259 |

Spatial giảm macro loss **0,0907%** so frozen; temporal giảm **0,2707%**.
Temporal thấp hơn spatial khoảng **0,18%**, cải thiện cùng chiều ở cả ba
domains nhưng mức chênh **rất nhỏ**, chỉ một seed. Không coi đây là bằng chứng
temporal PAD tốt hơn.

- Hai variants đều hoàn tất 360 steps, pause/resume thành công ở bước 180;
  frozen backbone hashes không đổi và giống nhau, adapter weights có thay đổi.
- Mỗi arm dùng **718 clip draws từ pool 359 fitting originals**. Sampling có
  replacement: thực tế thấy **219 unique originals** (CASIA36, MSU18, SiW165),
  giống nhau cho hai arms; không claim mọi fitting video đã được train.
- Validation bao phủ đủ **117 originals** ở mỗi epoch. Checkpoint hashes,
  source-only bindings và actual epoch sampling commitments đã qua postflight.
- Separate-process best-EMA replay trên ba originals/arm có maximum profile
  error **0**; đây là measured replay, không claim deterministic training.
- Tổng job **1.760,66 giây (~29,34 phút)**, sampled device-wide GPU peak
  **5,748 GiB**, worker-tree CPU RSS peak **4,124 GiB**, new output **~3,10 MiB**.
  Own PyTorch reserved peaks: spatial **2,461 GiB**, temporal **2,473 GiB**.
  Không chạm cap; không automatic retry.

Training epoch-mean losses tăng từ khoảng 0,01027 lên 0,01235 (spatial) và
0,01023 lên 0,01224 (temporal), trong khi fixed validation loss giảm nhẹ.
Epochs dùng noise/timesteps/windows khác nhau, nên hai training means không
phải một fixed-probe convergence curve. Chưa claim đã hội tụ, cũng không quy
nguyên nhân cho EMA hoặc learning rate khi chưa ablation.

## Approval và protocol khóa trước chạy

Người dùng duyệt pilot sau [source preparation 006](006-explicit-source-continuation.md):
**45 phút cho toàn bộ job, GPU memory 8 GiB, output mới 2 GiB**, hai variants
chạy tuần tự trong conda env `fas`. Không cấp quyền raw decoding, attack/OULU
model forwards, calibration hoặc target evaluation.

Plan SHA-256:
`b10511fd7e09e1c217f67985445dc294b4cac446333c2f1009c1f05ba9239cf0`.

Input prepared-manifest SHA-256:
`c2aa3ead64981a863fce6a78339a929c6d037f43609a9ff6a36988a183b84a7d`.
Local bundle SHA-256:
`87e6b56cea4c1d7d803579425b7219b0cc06914878340c038ea33eecec5b3fc6`.
Không tải weights hoặc data mới.

## Matched comparison

| Thành phần | Spatial adapter | Temporal adapter |
|---|---|---|
| Backbone | Cùng pinned pretrained DDPM | Cùng pinned pretrained DDPM |
| Backbone update | Frozen | Frozen |
| Temporal kernel của residual adapter | 1 | 3 |
| Adapter channels | 32 | 32 |
| Fitting/validation | 359 / 117 source bonafide originals | Giống hệt |
| Clip | Random 2 giây, T16, RGB256, 8 FPS | Giống hệt |
| Epochs/optimizer steps | 2 / 360 | 2 / 360 |
| Effective batch / clip microbatch | 2 / 1 | 2 / 1 |
| Frozen backbone frame microbatch | 4 | 4 |
| Optimizer | AdamW, LR 1e-4 rồi 5e-5 | Giống hệt |
| EMA | 0,999 | 0,999 |
| Numeric | FP32, không AMP/TF32 | Giống hệt |

Đây là matched backbone/data/noise/compute-budget comparison, không phải
parameter-count-identical ablation: temporal kernel có thêm parameters.
Pretrained backbone đã học external face images; **không** gọi đây là
bonafide-only training từ initialization.

Sampling cân bằng source domains; các windows không được đếm như independent
videos. Epoch 0 lấy CASIA/MSU/SiW lần lượt 120/119/120 draws; epoch 1:
119/120/120. Cùng cache hashes, seed, addressed windows, augmentation, noise và
timesteps cho hai variants. Mỗi epoch ghi sampling commitment và worker phải
khớp locked schedule.

Validation dùng fixed middle window của đủ 117 originals, ba noise levels
alpha-bar 0,9/0,5/0,1, hai addressed draws/level. Báo cáo mean từng domain và
macro trung bình ba domain; không để 99 SiW validation videos lấn át CASIA12/MSU6.
Frozen/zero-adapter baseline được đo trước update. Best EMA chọn bằng source
macro validation loss, cùng selection rule cho hai variants; last EMA cũng giữ
để xem tác động chọn checkpoint.

## Checkpoints và reproducibility

[Training loop](../src/diffusionfas/training.py) hỗ trợ explicit pause sau
epoch, vẫn ghi checkpoint `complete: false`. Pilot pause sau epoch 1 (180 bước),
load optimizer/EMA/RNG và chạy tiếp epoch 2; chỉ final last checkpoint mới được
claim completed. Best checkpoint có thể thuộc epoch 1, không dùng nó làm bằng
chứng toàn bộ run hoàn tất.

Đã reproduce và sửa CUDA resume bug trước real-data run: checkpoint load với
CUDA map-location làm CUDA RNG state thành GPU tensor, nhưng
`torch.cuda.set_rng_state_all` cần CPU ByteTensor. Bản sửa đưa riêng các RNG
states về CPU; không đổi optimizer/EMA placement hoặc sampling seed.

Worker kiểm tra frozen-backbone hash không đổi và adapter weights có thay đổi.
Sau hai variants, một **process mới** load best EMA checkpoints và replay ba
validation originals cố định (mỗi domain một), so residual profiles với values
đã ghi lúc validation: `rtol=1e-5`, `atol=1e-6`.
Không claim bitwise-deterministic GPU training hoặc đã so real-data resumed
trajectory với một full uninterrupted control run.

## Resource guards và privacy

[Runner](../scripts/adapter_pilot.py) dùng shared supervisor cho training,
validation, pause/resume và child replay; không cấp thêm 45 phút cho mỗi arm.
CPU worker-tree RSS được cap bảo thủ ở 8 GiB. Output cap cộng toàn bộ new root,
gồm checkpoints/logs/receipts, không chỉ model weights.

WSL không cung cấp usable per-process memory accounting qua compute-apps query
ở lần kiểm tra này. Watchdog vì vậy đọc **device-wide memory.used** bằng
`nvidia-smi`: nếu toàn device vượt 8 GiB thì chỉ dừng owned pilot process group,
không tác động process của người khác. Đây là conservative guard, gồm cả GPU
usage của ứng dụng khác; không báo device usage thành own-process usage.
PyTorch allocator riêng được giới hạn tối đa 7 GiB, chừa 1 GiB cho overhead.
Watchdog nominal interval 0,2 giây cộng telemetry-call time; không claim
instantaneous OS GPU quota.

Inputs, crops, individual validation profiles, membership, checkpoints và logs
giữ private. Public evidence chỉ export domain aggregates, opaque commitments
và resource measurements. Partial-run receipts vẫn ghi arms đã bắt đầu/hoàn
tất, không biến failure thành zero training hoặc success.

## Validation

[Validation v10](../results/validation-v10.json): **76 tests passed, 0 skipped**,
gồm actual CUDA checkpoint-resume và actual pinned FFmpeg helper trên synthetic
media. Các regressions bổ sung kiểm tra locked pilot scope/sampling/caps,
incomplete-preparation rejection, changed manifests, forbidden target permission,
private-score filtering, partial-run accounting và GPU watchdog.

Portable tests sẽ skip riêng CUDA-resume integration khi không có CUDA; lần
validation này có GPU và test đã thực thi. Không claim Pylance/editor type-check.

## Cách chạy sau explicit approval

Đây là fixed bounded pilot, không tự mở rộng epochs/budget. Dùng root mới và
giữ prepared cache roots cũ để manifest references còn hợp lệ:

```bash
conda run --no-capture-output -n fas python scripts/adapter_pilot.py plan \
  --manifest /path/to/private/prepared-manifest.json \
  --manifest-sha256 <approved-prepared-manifest-sha256> \
  --preparation-plan /path/to/private/completed-preparation/plan.json \
  --bundle /path/to/private/verified-bundle \
  --bundle-sha256 <approved-bundle-manifest-sha256> \
  --private-root /path/to/private/new-pilot

conda run --no-capture-output -n fas python scripts/adapter_pilot.py run \
  --plan /path/to/private/new-pilot/plan.json

conda run --no-capture-output -n fas python scripts/adapter_pilot.py report \
  --plan /path/to/private/new-pilot/plan.json \
  --out /path/to/new-aggregate-pilot.json
```

## Giới hạn diễn giải

Pilot một seed chỉ đánh giá optimization, source-bonafide denoising loss,
checkpoint correctness và compute cost. Loss thấp hơn **không chứng minh**
phân biệt spoof tốt hơn hoặc generalize tốt hơn sang lighting/camera mới.
Chưa fit normalization/threshold, chưa có APCER/BPCER/ACER hay target AUC.
Strict source BPCER 5% calibration blocker của CASIA/MSU vẫn giữ nguyên.

Chưa tự train dài hơn hoặc mở attack/target access. Thí nghiệm tiếp theo cần
budget/protocol riêng, đánh giá thêm training coverage và held-out source spoof
separation thay vì chọn model chỉ bằng bonafide denoising loss; không dùng OULU
để chọn hyperparameters hoặc sửa calibration policy.
