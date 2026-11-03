# diffusionfas

Offline-first bonafide-only short-clip PAD research.

- [Kế hoạch ban đầu](docs/001-bonafide-only-short-clip-diffusion-plan.md).
- [Implementation, preflight thực tế và những gì còn bị chặn](docs/002-offline-implementation-and-preflight.md).
- [Aggregate metadata/resource preflight](results/preflight-v2.json).

**Chưa tải pretrained checkpoint, chưa training trên PAD videos và chưa đánh giá
target.** Default config đóng mọi quyền decode/scoring/training. Synthetic tests
không phải kết quả FAS.

## Môi trường và tests

Mọi lệnh Python chạy bằng **conda env `fas`**. Dependencies được khai báo trong
[pyproject.toml](pyproject.toml). Không tạo venv khác hoặc nâng cấp môi trường
đang phục vụ thí nghiệm khác.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run --no-capture-output -n fas python -m unittest discover -s tests -v

PYTHONPATH=src conda run --no-capture-output -n fas python -m diffusionfas --help

conda run --no-capture-output -n fas python scripts/validate.py \
  --out /path/to/new-validation-receipt.json
```

Trong lần triển khai này chỉ bổ sung package còn thiếu `diffusers==0.35.1` với
`--no-deps`; các package cũ không được nâng cấp. Không tải model weights.

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

Không có lệnh download hoặc unsafe pickle fallback. Loader chỉ đọc:

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
Upstream liệt kê legacy `.bin`; acquisition và safe conversion cần quyết định
riêng, chưa thực hiện. Model-card license không thay cho review quyền dữ liệu.

## Giới hạn hiện tại

- Chưa có actual multi-frame PAD crops; eligibility hiện chỉ từ saved metadata.
- Chưa benchmark denoiser thật trên GPU; không có estimated training runtime đáng tin.
- Source calibration bonafide CASIA/MSU quá ít cho strict 5% finite-sample policy
  đã chọn; không tự nới ngưỡng hoặc đổi split.
- Archive staging/extraction, hard process watchdog và full-matrix orchestration
  chưa được cho phép/chốt budget; không claim production execution readiness.
- SiW subject identities và các near-duplicate/capture-independence giới hạn cũ
  vẫn còn nguyên.
- Attack family không có trong source integration export hiện tại được báo
  `unknown`; không suy diễn subtype từ tên file.
