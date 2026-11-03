# diffusionfas

Offline-first bonafide-only short-clip PAD research.

- [Kế hoạch ban đầu](docs/001-bonafide-only-short-clip-diffusion-plan.md).
- [Implementation, preflight thực tế và những gì còn bị chặn](docs/002-offline-implementation-and-preflight.md).
- [Aggregate metadata/resource preflight](results/preflight-v2.json).
- [Checkpoint, GPU và source preprocessing probe](docs/003-bounded-checkpoint-gpu-and-source-probes.md).
- [Aggregate probe evidence](results/probe-v2.json).
- [Strict MSU decoder đã wire vào CLI/config](docs/004-strict-msu-decoder-policy.md).
- [Full source-bonafide preparation có budget](docs/005-bounded-full-source-preparation.md).

**Đã tải checkpoint private và chạy bounded engineering probes trong env `fas`;
chưa training trên PAD videos, chưa đánh giá target.** Full-source preparation
attempt đã dừng sau ~61 phút vì lỗi cache-QA: 282/491 originals prepared,
198 chưa thử và 1 interrupted; 9 quá ngắn, 1 tracking failure được giữ nguyên.
Đã sửa sampler và qua synthetic regressions, **chưa chạy lại, chưa sẵn sàng train**.
Xem [aggregate evidence](results/source-preparation-v1.json).
Default scientific config vẫn đóng mọi quyền decode/scoring/training.
Synthetic optimizer steps không phải kết quả FAS.

## Môi trường và tests

Mọi lệnh Python chạy bằng **conda env `fas`**. Dependencies được khai báo trong
[pyproject.toml](pyproject.toml). Không tạo venv khác hoặc nâng cấp môi trường
đang phục vụ thí nghiệm khác. [Validation mới nhất](results/validation-v6.json):
62 tests passed, gồm actual pinned FFmpeg helper trên video tổng hợp.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run --no-capture-output -n fas python -m unittest discover -s tests -v

PYTHONPATH=src conda run --no-capture-output -n fas python -m diffusionfas --help

conda run --no-capture-output -n fas python scripts/validate.py \
  --fas-root /path/to/fas \
  --out /path/to/new-validation-receipt.json
```

Chỉ bổ sung package còn thiếu `diffusers==0.35.1` với `--no-deps`; các package cũ
không được nâng cấp. Checkpoint chỉ được tải bằng explicit bounded acquisition
probe; không có automatic download trong scientific loader.

`--fas-root` trong validator bật native-helper integration test, chỉ dùng video
tổng hợp. Nếu không cung cấp source root (hoặc `DIFFUSIONFAS_FAS_ROOT`), portable
suite skip riêng test này; không cần PAD media/checkpoints.

## Lệnh metadata-only: có thể chạy trước training approval

Private manifests và individual records phải nằm ngoài Git worktree. Thay các
đường dẫn ví dụ bằng private paths thực tế.

```bash
PYTHONPATH=src conda run --no-capture-output -n fas python -m diffusionfas import-fas \
  --source-manifest /path/to/private/source_manifest.json \
  --partitions /path/to/private/partitions.json \
  --oulu-media-root /path/to/private/oulu_per_video_audit_v1/videos \
  --out /path/to/private/diffusionfas/metadata.json

PYTHONPATH=src conda run --no-capture-output -n fas python -m diffusionfas preflight \
  --manifest /path/to/private/diffusionfas/metadata.json \
  --config configs/offline-v1.json \
  --storage-root /path/to/storage \
  --out /path/to/new-aggregate-preflight.json
```

Importer pin source manifest/partition hashes, kiểm tra transaction/media receipts,
giữ split và SiW exclusions, bảo toàn original technical failures. Không đọc
pixel/video payload, không chạy detector/model.

## Scientific pipeline: chỉ chạy sau approval/budget mới

Các subcommands có `--help`; config hiện tại cố ý không cho chạy chúng:

1. `bind-media`: nhận explicit mapping original key -> extracted local file, kiểm
   tra SHA-256. Không tự giải nén archive hoặc chọn release khác.
2. `prepare`: source fitting/validation/calibration hoặc target evaluation theo
   gate riêng; native-frame hash checks, tracking, RGB256 caches.
3. `train`: bonafide source-only, video-balanced windows, EMA, early stopping,
   checkpoint/replay/resume và cooperative resource guards.
4. `profiles`: multi-noise source/target residuals với gates tách biệt; middle-frame
   baseline dùng cùng spatial checkpoint và cùng frame-addressed noise.
5. `calibrate`: fit robust normalization từ bonafide source validation; strict
   bonafide threshold hoặc attack-aware source calibration.
6. `evaluate`: áp dụng source policy đã khóa, không chọn threshold bằng target;
   all-original/usable metrics, oracle diagnostic và group bootstrap.
7. `feature-baseline`: local-only frozen DINO-Reg + one-class shrinkage Gaussian,
   mean features hoặc temporal difference statistics; dùng chung calibration/eval.

`prepare` cần checkout `fas` và detector cục bộ có hash được chỉ định; reuse
preprocessing đã có, không copy detector implementation. `feature-baseline` cần
checkout DINO/trọng số cục bộ được xác minh. Không tự tải các assets này.

### Decoder policy

[Default config](configs/offline-v1.json) chọn strict FFmpeg cho MSU, OpenCV cho
các domain còn lại; execution gates vẫn đóng. Với approved source-preparation
config sau này, `prepare` cần thêm `--source-manifest /path/to/accepted/source_manifest.json`
khi chọn native decoder. Manifest phải khớp imported provenance; helper code,
transaction/media receipts, timestamps và full native RGB sequence đều được bind.

Decoder error không fallback sang OpenCV và không đổi accepted hashes. Failed
originals giữ nguyên trong output manifest, kèm technical failure. Chi tiết
policy, diagnostics và command example ở [tài liệu 004](docs/004-strict-msu-decoder-policy.md).

### Full source preparation

[Standalone orchestrator](scripts/prepare_sources.py) có ba commands:
`plan` khóa selection/assets/code/caps; `run` thực thi dưới bounded supervisor;
`report` export aggregate evidence. Full eligible intervals được cache ở 8 FPS
và mỗi cache được QA bằng ba windows 2 giây. Một original được stage tại một
thời điểm, rồi xóa raw sau preprocessing.

Fold target OULU có 491 source bonafide fitting/validation originals, gồm 482
metadata-eligible và 9 quá ngắn; mọi failure vẫn nằm trong denominator. Không
thay video, bỏ hash check hoặc tự retry. Xem [tài liệu 005](docs/005-bounded-full-source-preparation.md)
về scope, finite budget, private outputs và cách chạy. Không tự chuyển sang train.

### Các model variants

| Variant | Trọng số ngoài | Tham số học |
|---|---|---|
| `pretrained_spatial` | Local safe bundle | Không; comparator frozen |
| `pretrained_spatial_adapter` | Local safe bundle | Adapter kernel thời gian 1 |
| `pretrained_temporal` | Local safe bundle | Adapter kernel thời gian 3 |
| `scratch_spatial` | Không | Spatial U-Net |
| `scratch_temporal` | Không | Spatial U-Net + temporal adapter |
| `scratch_fixed_noise` | Không | Fixed-noise denoising baseline |

Fixed-noise baseline phải đặt `scoring.alpha_bars` thành `[0.5]`; timestep
conditioning của backbone luôn là hằng số. Epsilon loss ở mức noise cố định
tương đương clean-image MSE nhân một hằng số.

Primary pretrained comparison phải là spatial-adapter vs temporal-adapter,
không chỉ frozen vs adapted. So khác resolution/schedule là exploratory,
không phải matched backbone ablation.

### Local pretrained bundle contract

Scientific loader không tự download hoặc dùng unsafe pickle fallback. Explicit
acquisition probe tách biệt được mô tả trong tài liệu 003. Loader chỉ đọc:

```text
bundle/
  manifest.json
  config.json
  scheduler_config.json
  model.safetensors
```

```json
{
  "upstream_revision": "<exact upstream commit>",
  "rights_reviewed": true,
  "files": {
    "config.json": "<sha256>",
    "scheduler_config.json": "<sha256>",
    "model.safetensors": "<sha256>"
  }
}
```

Config cần `model.local_bundle` và `model.bundle_manifest_sha256`. Native
resolution và scheduler lấy từ bundle, không ghi đè bằng scratch settings.
Candidate hiện tại là `google/ddpm-celebahq-256`, revision được pin trong config.
Legacy `.bin` đã được pin byte count/SHA-256, load với `weights_only=True` và
chuyển lossless sang private safetensors. Model-card license không thay cho
review quyền dữ liệu/commercial use.

## Giới hạn hiện tại

- Engineering probe chỉ có 6 source originals, không chứng minh tracking success
  toàn dataset hoặc khả năng phân biệt bonafide/spoof.
- GPU timing chỉ từ tensors tổng hợp; chưa đo end-to-end training.
- MSU strict FFmpeg đã được wire vào CLI/config, không automatic fallback.
  Config cũ không có decoder policy giữ legacy OpenCV behavior; không dùng
  config legacy đó cho MSU full preparation.
- Source calibration bonafide CASIA/MSU quá ít cho strict 5% finite-sample policy
  đã chọn; không tự nới ngưỡng hoặc đổi split.
- Exact-member staging và sampled process watchdog đã kiểm tra cho probe nhỏ.
  Full-source preparation, scientific training và full matrix chưa được chốt
  budget; không claim production execution readiness.
- SiW subject identities và các near-duplicate/capture-independence giới hạn cũ
  vẫn còn nguyên.
- Attack family không có trong source integration export hiện tại được báo
  `unknown`; không suy diễn subtype từ tên file.
