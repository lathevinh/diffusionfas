# 003 - Checkpoint, GPU và source preprocessing probes

Snapshot của phase probes. Giới hạn CLI native-decoder bên dưới đã được giải
quyết ở [phase 004](004-strict-msu-decoder-policy.md); số đo/probe receipts giữ nguyên.

## 1. Kết luận và ranh giới

**Engineering feasibility đã qua:** checkpoint dùng được, synthetic optimization/
scoring/reload chạy được trên RTX 4080 SUPER, và 12/12 cửa sổ định trước từ 6
source bonafide originals tạo được tracked crops có native RGB binding.

**Chưa có bằng chứng về PAD accuracy hoặc cross-domain generalization.** Không
train model trên PAD crops, không dùng attack videos, không decode OULU target,
không chọn threshold hoặc checkpoint từ target. Mọi Python/tests/probes dùng
conda env `fas`; không cài thêm/nâng cấp package trong phase này.

Evidence:

- [Aggregate receipts cuối cùng](../results/probe-v2.json).
- [43 tests và implementation hashes](../results/validation-v3.json).
- [Script chạy từng bounded phase](../scripts/probe.py).
- [Script kiểm tra lineage và xuất aggregate](../scripts/summarize_probe.py).

Weights, original IDs, membership, media paths, face crops, decoder logs và
per-window receipts nằm ngoài Git. Public results không chứa các dữ liệu này.
Report probe-v1 là aggregate trung gian; probe-v2 bổ sung actual native-vs-
microbatched CUDA verification. Receipt validation-v2 là snapshot trước chỉnh
reproducibility của summarizer; validation-v3 bind implementation cuối.

## 2. Checkpoint đã thực sự tải và chuyển đổi

| Thuộc tính | Giá trị |
|---|---|
| Repository | `google/ddpm-celebahq-256` |
| Revision | `cd5c944777ea2668051904ead6cc120739b86c4d` |
| Upstream weight bytes | 454.853.117 |
| Upstream SHA-256 | `efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d` |
| Parameters | 113.673.219 |
| Native resolution/schedule | 256, linear DDPM, 1.000 timesteps |
| Bundle manifest SHA-256 | `87e6b56cea4c1d7d803579425b7219b0cc06914878340c038ea33eecec5b3fc6` |

[Acquisition](../src/diffusionfas/acquisition.py) pin text assets bằng Git blob
hash, weight bằng LFS SHA-256 và exact byte count. Chỉ load bằng
`torch.load(weights_only=True)`; không có unsafe fallback.

48 attention tensor keys được đổi tên sang Diffusers API hiện tại; không tensor
nào cần reshape trong checkpoint thật. Strict key coverage, strict state load
và exact tensor-value equality đã kiểm tra trước khi lưu private safetensors.
Acquisition/conversion hoàn thành khoảng 30,47 giây.

Model card khai báo `apache-2.0`. Review chỉ xác nhận khai báo này cho private
research probe, không chứng nhận quyền unrestricted/commercial đối với dữ liệu
CelebA-HQ. Không redistribute weights hoặc tải ảnh mẫu.

## 3. GPU: số đo thực tế, không phải dự báo accuracy

Input tổng hợp `[1, 16, 3, 256, 256]`, FP32, không AMP/TF32. Frozen U-Net xử lý
microbatch 4 frames; adapter vẫn nhận **đủ 16 frames**, không cắt temporal context.
Adapter channels = 32. Các arm dùng cùng checkpoint/native scheduler.

| Arm | Trainable params | Mean step, giây | Score 3 levels x 2 draws, giây | Peak allocated, GiB | Peak reserved, GiB |
|---|---:|---:|---:|---:|---:|
| Frozen spatial | 0 | 0,328 | 1,993 | 1,81 | 2,45 |
| Spatial adapter, temporal kernel 1 | 46.051 | 0,331 | 1,973 | 1,81 | 2,46 |
| Temporal adapter, temporal kernel 3 | 48.099 | 0,333 | 1,975 | 1,81 | 2,46 |

Mỗi learned arm chạy 4 synthetic optimizer steps: 1 warmup và 3 measured steps.
Frozen arm chỉ forward/loss, không optimizer. GPU timing có synchronize.
Đây không phải epoch trên bonafide videos, không phải training throughput bao gồm
I/O/augmentation/validation và không chứng minh convergence.

Đã xác nhận:

- Zero-init adapters có max difference 0 so với frozen prediction.
- Loss/gradient hữu hạn; mỗi learned arm nhận nonzero gradients.
- Backbone tensor hashes không đổi sau synthetic updates.
- Adapter safetensors save/load, và cả 3 arm replay trong **process CUDA khác**:
  max difference 0 với tolerance `rtol=1e-5, atol=1e-6`.
- Actual pretrained U-Net forward full 16 frames so với microbatch 4:
  max absolute difference `2,145767e-6`; kiểm tra dưới
  `rtol=1e-4, atol=1e-5` và absolute cap `1e-4`.
- Riêng full-16 verification peak allocated khoảng 5,47 GiB, reserved 8,32 GiB.
  Không trộn các peak này với bảng benchmark microbatch-4.

CPU regression ban đầu lộ tolerance quá sát: full-suite microbatch comparison
lệch tối đa `1,186971e-6`, vượt absolute tolerance `1e-6` gần giá trị 0. Đã đo thêm
12 fixed seeds: max differences khoảng `1,04e-6` tới `1,67e-6`; test dùng seed cố
định và `rtol=1e-5, atol=2e-6`. Không claim equivalence bitwise hoặc deterministic
CUDA training. Đây là sai khác numerical nhỏ do đổi batch shape, không đổi input
frames hoặc temporal adapter context.

Spatial/temporal adapters gần cùng capacity nhưng không exact parameter-matched.
Các số này chỉ chốt **có thể chạy**, không chứng minh temporal adapter hữu ích.

## 4. Source preprocessing: selection khóa trước outcomes

[Private selection](../src/diffusionfas/source_probe.py) được ghi trước detector:
seed 20261103, lấy minimum addressed hash trong eligible bonafide population của
mỗi source/partition. Không thay split hoặc chọn lại video sau khi thấy kết quả.

- CASIA/MSU/SiW: mỗi domain 1 fitting + 1 validation original.
- Mỗi original: middle 2s và một addressed random 2s window.
- Tổng 6 originals, 12 fixed windows; không cặp window nào trùng native indices.
- Mỗi window 16 distinct frames ở 8 FPS; output RGB `[16, 256, 256, 3]`.
- Selection commitment:
  `fc49a5bd03375455b48eee49a6b6ffc7a678fe8848b4cc10b71b4b1d175edd3a`.

| Source | Originals | Initial success | Explicit native follow-up | Final fixed windows passed |
|---|---:|---:|---:|---:|
| CASIA | 2 | 4/4 | 0 | 4/4 |
| MSU | 2 | 0/4 | 4/4 | 4/4 |
| SiW | 2 | 4/4 | 0 | 4/4 |

**Không che initial MSU failure.** OpenCV decode không khớp accepted RGB hashes;
cả 4 windows được ghi technical failure, không phải anomaly scores. Sau đó
explicitly dùng strict FFmpeg replay có sẵn từ checkout `fas`, giữ nguyên 2
originals/4 windows/accepted receipts và hash criteria. Cả hai decode passes của
mỗi window được kiểm tra đủ full-video RGB sequence: 8 passes, tổng 2.168 frame
verifications. Không chỉ kiểm tra frame đầu hoặc waive decoder differences.

Tổng cộng **16 window attempts**, không phải 16 independent samples: 12 ban đầu
và 4 native repair attempts trên đúng các windows MSU đã chọn.

Initial preparation khoảng 53,18 giây; native follow-up khoảng 21,54 giây.
Initial invocation của native follow-up còn lỗi import `fas` trước khi staging/
decoding. Đã sửa import path, chạy lại bằng explicit attempt label, giữ cả failed
và successful supervision receipts. Không có automatic retry hoặc replacement.

Staging chỉ exact selected members, kiểm tra byte size + media SHA-256:

- CASIA: existing `bsdtar`, không bulk extraction.
- SiW: exact ZIP members và CRC.
- MSU: reuse seekable virtual concatenation của 16 split volumes trong 8 outer
  ZIPs, không tạo joined archive. Seek có thể decompress unrelated container
  bytes; **không decode unrelated video**.

Newly staged originals đã xóa sau xử lý; private crops/receipts được giữ. SCRFD
chạy CPU, reuse detector/crop của `fas` với bound hashes. PAD crops không đi qua
diffusion model và không tạo PAD scores.

### Decoder policy chưa được âm thầm đổi

`prepare_video(..., native_replay=...)` hỗ trợ explicit strict replay và vẫn
kiểm tra hash/count; default OpenCV behavior được giữ để tránh âm thầm đổi
scientific contract. Probe đã sử dụng native hook cho MSU.

**Ở snapshot 003, CLI `prepare` chưa wire native hook.** Việc này đã được hoàn
thành trong [004](004-strict-msu-decoder-policy.md), giữ all-original failure
accounting; không sửa accepted hashes, đổi video hoặc bỏ những video khó.

## 5. Resource supervision và reproducibility

[Supervisor](../src/diffusionfas/supervision.py) chỉ terminate process group tự
tạo, không tác động workload khác. Mỗi phase chạy process riêng, private logs
và immutable receipts; resource failures không trả success.

| Phase | Wall cap, giây |
|---|---:|
| Fixed selection plan | 180 |
| Acquisition/conversion | 900 |
| Synthetic GPU arms | 600 |
| Initial source probe | 1.200 |
| Native source follow-up | 600 |
| Separate-process replay | 300 |
| Native frame-microbatch comparison | 120 |

Process-tree RSS cap 12 GiB; owned private-output cap 8 GiB; sampled mỗi 0,2 giây.
Termination grace tối đa 5 giây. Đây là sampled watchdog, **không phải**
instantaneous OS-enforced RSS/disk quotas. Own PyTorch CUDA allocator giới hạn
khoảng 10,4 GiB ở snapshot hiện tại, vẫn chừa VRAM cho workload khác; không giới
hạn hoặc claim đo toàn bộ CUDA driver allocation.

Reproduce bằng private output root mới. Chạy riêng từng phase qua
`conda run --no-capture-output -n fas python scripts/probe.py <phase>`, cung cấp
`--private-root`, `--fas-root`, `--metadata`, `--source-manifest`, `--detector`.
Thứ tự: `plan`, `acquire`, `gpu`, `source`, `source-native`, `replay`, `microbatch`.
`source-native` chỉ chấp nhận đúng 4 MSU RGB failures của initial report.
Không overwrite phase outputs; explicit `--attempt-label` chỉ đổi log/receipt
name, không tự cho phép overwrite model/crop artifacts.

```bash
conda run --no-capture-output -n fas python scripts/validate.py \
  --out /path/to/new-validation-receipt.json

conda run --no-capture-output -n fas python scripts/summarize_probe.py \
  --private-root /path/to/private/probe-root \
  --out /path/to/new-aggregate-probe.json
```

Default scientific execution flags vẫn toàn bộ false; không dùng probe để mở
source training, target decode hoặc target evaluation.

## 6. Bước tiếp theo, chưa tự chạy

1. Chốt/wire native decoder policy vào full preparation CLI/config và mở rộng
   source-only compatibility coverage có fixed membership, không chọn theo score.
2. Xin budget full source-bonafide preparation cho outer fold target OULU.
   Metadata trước tracking có 361 fitting và 121 validation originals; successful
   crop counts có thể thấp hơn, phải report nguyên denominator/failures.
3. Sau khi chốt data contract/budget, mới chạy spatial-adapter vs temporal-adapter
   pilot, learned parameters và checkpoints chọn bằng source bonafide validation.
   DINO one-class và scratch controls chưa được benchmark trong phase này.
4. Chốt calibration policy trước attack/target evaluation. CASIA 12 và MSU 6
   bonafide calibration originals vẫn không đáp ứng **policy đã chọn**: strict
   BPCER 5% với ít nhất một rejection unit. Không đổi split, nới rate hay tự chọn
   accept-all. Đây là giới hạn finite-sample policy, không phải chứng minh mọi
   empirical small-sample operating point đều bất khả thi.

Kết quả hiện tại đủ để bỏ nghi ngờ “checkpoint này có chạy nổi trên GPU không”,
chưa đủ để kết luận diffusion tốt hơn transformer hoặc tổng quát hóa tốt.
