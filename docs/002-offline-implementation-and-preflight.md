# 002 - Offline implementation và preflight thực tế

Ngày: 2026-11-03.

Đây là snapshot của phase offline ban đầu. Checkpoint/GPU/source probes đã được
cho phép và thực hiện sau đó; xem [báo cáo 003](003-bounded-checkpoint-gpu-and-source-probes.md).

**Trạng thái: đã triển khai code và kiểm tra offline/synthetic; chưa chạy scientific
training, chưa tải checkpoint, chưa đánh giá target.**

## 1. Quyết định được xác nhận

Sau [kế hoạch 001](001-bonafide-only-short-clip-diffusion-plan.md), người dùng chọn:

- Ưu tiên pretrained spatial denoiser + temporal adapter.
- Scratch diffusion làm đối chứng, không mặc định dữ liệu đủ để train từ đầu.
- Đợt này chỉ implementation, tests, metadata/resource preflight.
- Chưa cho phép tải checkpoint hoặc training thật trước khi chốt budget.
- Mọi Python/install/test commands dùng conda env `fas`.

Tài liệu này mô tả implementation thực tế và các điều chỉnh so với đề xuất.
Không coi một test optimizer step trên dữ liệu tổng hợp là PAD training result.

## 2. Code đã có

- [Metadata import](../src/diffusionfas/fas_import.py): pin source revision,
  manifest/partition hashes, source-use exclusions và transaction/media bindings.
- [Data contracts](../src/diffusionfas/data.py): isolation checks, native timestamp
  sampling, addressed random training windows, fixed validation/test windows.
- [Preprocessing](../src/diffusionfas/preprocessing.py): explicit local originals,
  native RGB checks, reuse SCRFD/crop từ checkout `fas`, bounded gap interpolation,
  continuity guards, smoothing và RGB256 cache.
- [Clip dataset](../src/diffusionfas/dataset.py): verify cache, video/source-balanced
  sampling, consistent clip augmentation.
- [Models](../src/diffusionfas/models.py): local pretrained denoiser, spatial/temporal
  residual adapters, scratch U-Net và fixed-noise baseline.
- [Training](../src/diffusionfas/training.py): source-only fitting, EMA, source
  validation, accumulation, early stopping, checkpoint/resume và cooperative caps.
- [Scoring](../src/diffusionfas/scoring.py): native-scheduler multi-noise residual,
  frame-addressed reproducible noise và equal-source robust normalization.
- [Baselines](../src/diffusionfas/baselines.py): frozen local DINO-Reg,
  mean/temporal-difference features, source-only shrinkage Gaussian.
- [Metrics](../src/diffusionfas/metrics.py): strict/attack-aware source thresholds,
  integer counts/ties, all-original/usable denominators, unknown-family reporting,
  oracle diagnostic và grouped paired-bootstrap helper.
- [CLI](../src/diffusionfas/cli.py): wiring từ metadata import tới locked evaluation.

Default [config](../configs/offline-v1.json) đóng source/target decode, source
training/scoring, target evaluation và download permissions; budgets để null.
Không có automatic download path.

## 3. Điều chỉnh có chủ đích so với 001

### 3.1. Pretrained candidate và native settings

Metadata/config công khai của `google/ddpm-celebahq-256` đã được kiểm tra:

- Revision `cd5c944777ea2668051904ead6cc120739b86c4d`.
- Model-card license `apache-2.0`.
- Pixel-space UNet2D, native resolution 256.
- Native DDPM schedule linear, 1.000 steps.
- Repository liệt kê legacy `.bin`, không phải sẵn một safe bundle.

Chỉ đọc metadata/config dạng text. Không tải weight, ảnh mẫu hoặc dữ liệu PAD.
Loader yêu cầu local safetensors, full SHA-256 commitments và explicit rights
review. Acquisition/conversion vẫn bị chặn.

Scratch feasibility config là 128/cosine. Không ép pretrained candidate xuống
128/cosine rồi coi như native behavior được bảo toàn. Cross-family comparison
này chưa matched; cần quyết định matched 256 configuration trước full study.

### 3.2. Adapter ngoài U-Net thay vì internal hooks

Adapter nhận noisy RGB và frozen predicted epsilon, encode không gian xuống 1/4,
thêm timestep embedding, convolution thời gian và zero-init residual epsilon.
Không sửa private internal blocks của Diffusers, không fine-tune pretrained
backbone trong bản đầu.

Đây là deviation so với ý tưởng chèn module tại nhiều internal feature scales.
Lý do: giữ API ổn định, dễ replay, nhỏ hơn và kiểm chứng rõ frozen backbone.
Không khẳng định nó mạnh bằng internal temporal adaptation.

Thêm **spatial adapter kernel 1** làm đối chứng học được. Temporal adapter kernel
3 phải so với đối chứng này; chỉ hơn frozen denoiser không cô lập temporal benefit.
Normalization của adapter thực hiện theo từng frame để kernel-1 control không
vô tình trộn thống kê qua thời gian.

Scratch U-Net ban đầu không spatial attention, nhằm giữ feasibility implementation
đơn giản. Không gọi đó là exact architecture của pretrained candidate.

### 3.3. Tracking và decoding

Reuse largest-face detection đã có, thêm continuity/scale guards và bounded
interior-gap interpolation. Không có face re-identification hoặc chứng nhận
identity tracking trên multiple-person video.

Preparation dùng hai decode passes để không giữ toàn bộ native-resolution
frames trong RAM: pass đầu lấy landmarks, pass sau crop theo track đã làm mượt.
Native selected-frame RGB hashes và full frame count phải khớp receipt; decoder
khác không được waiver ngầm. Actual PAD codec compatibility chưa chạy.

Cache giữ cả requested 8-FPS grid và native timestamps; không nội suy pixel.
Random windows chọn trên requested grid để native timestamp jitter không tạo
sampling failures giả, còn native timestamps phải nằm trong tolerance đã khóa.

`bind-media` yêu cầu original đã được stage cục bộ và có đúng hash. Không tự
extract ZIP/TAR/RAR trong đợt này. Archive staging cần budget và authority riêng.

### 3.4. Training và guards

Bản trainer hiện hỗ trợ FP32 rõ ràng, chưa AMP/BF16. EMA chỉ giữ trainable
parameters, không clone toàn bộ frozen backbone.

Synthetic test đã kiểm tra checkpoint reload và epoch-boundary resume cho kết
quả bằng nhau. Chưa claim bitwise-deterministic CUDA training hoặc exact resume
giữa optimizer step bị ngắt.

Resource guards là cooperative checks, không thay thế hard process supervisor.
Hard watchdog/full-matrix controller cần chốt trước scientific execution.

## 4. Preflight dữ liệu thực tế

Nguồn: accepted private source manifest và official-partition contexts của
`lathevinh/fas`, source revision
`707e3ddf6103835c55e937cbbd84081d91f188c7`.

Đã import **7.505 active originals**, kế thừa năm SiW source exclusions. Có một
OULU pre-detector failure không có transaction path; importer giữ nguyên failure,
không tạo record/frame giả để vượt qua.

Đã verify media receipt bindings cho các originals có receipt. Đây là đọc saved
JSON evidence, không decode lại media hoặc chạy detector.

### 4.1. Số video đủ sampling trước tracking mới

| Window | Eligible originals | Ineligible originals |
|---|---:|---:|
| 1 giây | 7.495 | 10 |
| 2 giây | 7.397 | 108 |
| 3 giây | 7.379 | 126 |

Ở 2 giây: 98 thiếu duration, 8 historical decode failures, 1 historical
pre-detector failure, 1 không đáp ứng timestamp sampling tolerance.

Không đổi window hoặc nới tolerance để cứu mẫu. Đây không phải số crop thành
công; tracking và native RGB replay mới vẫn có thể làm giảm support.

### 4.2. Bonafide source support cho 2 giây

| Source dataset | Fitting | Validation | Calibration |
|---|---:|---:|---:|
| OULU-NPU | 359 | 144 | 126 |
| CASIA-FASD | 36 | 12 | 12 |
| MSU-MFSD | 18 | 6 | 6 |
| SiW-Mv2 | 307 | 103 | 101 |

Mỗi outer fold bỏ toàn bộ dòng của target:

| Held-out target | Bonafide fitting từ 3 nguồn |
|---|---:|
| OULU-NPU | 361 |
| CASIA-FASD | 684 |
| MSU-MFSD | 702 |
| SiW-Mv2 | 413 |

Đây là bằng chứng cụ thể cho việc không đặt cược scratch diffusion là hướng
chính. Random windows tăng coverage trong video, không tăng số video độc lập.

### 4.3. Calibration blocker

Strict BPCER 5% implementation yêu cầu tối thiểu 20 bonafide calibration videos
mỗi source để có một empirical rejection unit ở mức này.

CASIA có 12, MSU có 6 trước tracking, nên **cả bốn folds đều có source không đủ
support cho strict 5% policy đã chọn**.

Không tự đổi thành accept-all, đổi split, thêm calibration windows như independent
samples hoặc hạ chuẩn để tạo kết quả. Cần phản biện policy/endpoint trước pilot
khoa học. Đây là giới hạn finite-sample policy đã chọn, không phải định lý rằng
không thể báo bất kỳ empirical operating point nào với sample ít.

## 5. Resource preflight

Đã kiểm tra bằng runtime trong conda env `fas`:

- Python 3.12.14, Torch 2.7.1, CUDA khả dụng.
- Một NVIDIA GeForce RTX 4080 SUPER, khoảng 16 GiB VRAM.
- Tại snapshot preflight: khoảng 14,68 GiB VRAM trống.
- CPU count 16; khoảng 847 GiB free tại storage root đã chọn.
- FFmpeg/FFprobe có sẵn.
- Bổ sung duy nhất `diffusers==0.35.1` với `--no-deps`; không nâng package cũ.

VRAM/disk trống là snapshot, không reservation hoặc throughput measurement.
Chưa tải pretrained weights nên chưa có actual denoiser GPU-memory/latency
benchmark. Không dùng snapshot này để hứa thời gian 12/36/full-matrix trainings.

Pylance MCP không kết nối được trong phiên này. Kiểm chứng thực thi dùng env
`fas`, unittest, actual Diffusers tiny networks và synthetic media; không claim
đã có Pylance diagnostics/type-check.

Aggregate evidence:

- [Preflight v1](../results/preflight-v1.json): snapshot đầu tiên.
- [Preflight v2](../results/preflight-v2.json): thêm explicit calibration blockers.

Private imported records/timestamps/hashes nằm ngoài Git, không public cùng report.

## 6. Validation và ranh giới kết luận

Tests bao phủ:

- Timestamp sampling, no frame duplication/padding và fixed/random windows.
- Group/subject/exact-content isolation, private-output location, immutable JSON.
- Bounded track interpolation, endpoint/jump failures.
- Actual synthetic FFV1 video decode -> track -> RGB256 cache.
- Real Diffusers tiny scratch forward/backward.
- Frozen backbone, zero-init adapter identity và adapter optimization.
- Local safetensors bundle verification, no automatic download.
- Reproducible multi-noise scoring và matched frame-addressed noise.
- Synthetic bonafide source training, EMA/checkpoint replay và exact epoch resume.
- Source-only calibration, ties, failures/denominators và grouped uncertainty.
- Fixed-noise baseline và Gaussian feature baseline.
- Source policy -> synthetic target metrics integration.

Số tests, kết quả và implementation hashes được ghi bằng actual test runner tại
[validation receipt](../results/validation-v1.json), không ghi tay từ kỳ vọng.

Synthetic images/features/video/labels được tạo trong tests và xóa cùng temporary
directories. Không dùng private PAD pixels cho test optimizer steps.

## 7. Những gì chưa được xác minh hoặc chưa được chạy

1. Actual PAD multi-frame detector/tracking/crop acceptance.
2. Pretrained candidate weight acquisition, rights review và safe conversion.
3. Quality của pretrained residuals với PAD, native GPU footprint/throughput.
4. Actual source training, checkpoint selection hoặc target scoring.
5. Strict calibration amendment cho source samples ít.
6. Archive extraction/staging và hard resource supervisor.
7. Full four-fold/multi-seed controller và matched-resolution full study.
8. Near-duplicate/capture independence và SiW subject identity.
9. Attack-family joins: export hiện tại không có subtype được xác minh, nên
   `unknown` được báo rõ thay vì suy diễn từ video IDs.

## 8. Bước tiếp theo cần chốt sau preflight

Ưu tiên một pilot context giữ OULU:

- Sources có 361 bonafide fitting videos trước tracking.
- Primary learned comparison: pretrained spatial adapter vs temporal adapter.
- Frozen pretrained denoiser không có training run.
- Scratch spatial/temporal và fixed-noise controls chỉ mở trong budget đã duyệt.

Nếu chạy đầy đủ mọi arm hiện có: năm neural trainings mỗi fold, tức 20 mỗi seed
hoặc 60 cho ba seeds, chưa tính debug/smoke runs. Đây không phải authority để
chạy; số 12/36 trong 001 thuộc matrix cũ trước pretrained/matched-adapter amendment.

Cần quyết định calibration policy, pretrained acquisition và actual pilot budget
trước khi bật các permission flags. Chưa tự động đề xuất numerical hours/caps vì
actual pretrained throughput chưa được đo.
