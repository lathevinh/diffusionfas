# diffusionfas

Offline-first bonafide-only short-clip PAD research.

- [Kế hoạch ban đầu](docs/001-bonafide-only-short-clip-diffusion-plan.md).
- [Implementation, preflight thực tế và những gì còn bị chặn](docs/002-offline-implementation-and-preflight.md).
- [Aggregate metadata/resource preflight](results/preflight-v2.json).
- [Checkpoint, GPU và source preprocessing probe](docs/003-bounded-checkpoint-gpu-and-source-probes.md).
- [Aggregate probe evidence](results/probe-v2.json).
- [Strict MSU decoder đã wire vào CLI/config](docs/004-strict-msu-decoder-policy.md).
- [Full source-bonafide preparation có budget](docs/005-bounded-full-source-preparation.md).
- [Continuation đã hoàn tất source preparation](docs/006-explicit-source-continuation.md).
- [Matched spatial/temporal source-only pilot](docs/007-matched-source-adapter-pilot.md).
- [Source-validation spoof diagnostic: raw residual chưa tách tốt](docs/008-source-validation-spoof-diagnostic.md).
- [CelebA-HQ-style crop và private HTML review trước ablation](docs/009-celebahq-geometry-visual-review.md).
- [Audit dataset: crop phải nằm hoàn toàn trên pixels thật, không padding](docs/010-real-pixel-crop-dataset-audit.md).
- [MSU preview không padding: 6 videos, dừng chờ duyệt ảnh](docs/011-msu-real-pixel-crop-review.md).
- [Frozen DDPM crop A/B trên cùng 22 MSU clips](docs/012-msu-frozen-crop-ablation.md).
- [MAT pretrained face inpainting: native-512 preview đã được duyệt](docs/013-mat-native512-inpainting-preview.md).
- [MAT masked reconstruction: protocol cố định trên 22 MSU validation clips](docs/014-msu-mat-anatomical-diagnostic.md).
- [Đối chiếu DINO–MAT: cùng 22 clips/T16, giữ crop đúng từng model](docs/015-matched-msu-dino-mat-comparison.md).
- [Audit sẵn sàng công bố: đóng góp, checklist sửa bài và shortlist tạp chí có điều kiện](docs/016-submission-readiness-audit.md).
- [DINO vùng mặt + GroupDRO: 6 ablations đã implement, chạy thật cần duyệt protocol/budget](docs/017-regional-dino-controlled-ablation.md).

**Đã hoàn tất source preparation và matched source-only adapter pilot trong env
`fas`, cùng source-validation development diagnostic; chưa có unbiased PAD
test hoặc target result.** Full-source preparation
đã hoàn tất sau explicit continuation approval: **476/491 originals usable,
gồm 359 fitting + 117 validation**. Giữ 9 short-video và 6 tracking failures
trong denominator; không còn interrupted/unattempted. Caches đã QA bằng 1.428
windows và verify lại toàn bộ hashes. Xem [aggregate evidence](results/source-preparation-v2.json)
và [lịch sử attempt đầu bị dừng](docs/005-bounded-full-source-preparation.md).
Hai adapter đã train 360 steps/model trên cùng sampling/noise schedule; source
validation macro MSE giảm 0,0907% (spatial) / 0,2707% (temporal) so frozen,
**chưa chứng minh spoof detection tốt hơn**. Xem [pilot evidence](results/adapter-pilot-v1.json).
Diagnostic đủ 283 validation originals, 259 usable / 24 failures; AUROC
frozen / spatial / temporal lần lượt: CASIA **0,3819 / 0,3819 / 0,3819**,
MSU **0,6765 / 0,6765 / 0,6765**, SiW **0,5006 / 0,5007 / 0,5012**.
Temporal macro chỉ **0,5199**, chưa cải thiện separation có ý nghĩa; **chưa
nên train dài hơn với raw residual recipe này**. Xem
[coverage, paired CIs và failure bounds](results/source-diagnostic-v2.json).
Đây là development evidence vì bonafide validation đã chọn checkpoints;
không fit threshold, đảo score, chạy calibration/target hoặc train thêm.
Lượt đầu dừng do staging guard được giữ trong
[failed-attempt evidence](results/source-diagnostic-v1.json); lượt sau chỉ chạy
sau explicit approval mới. Training dài hơn hoặc access mới vẫn cần
approval/budget riêng.
Default scientific config vẫn đóng mọi quyền decode/scoring/training.
Synthetic optimizer steps không phải kết quả FAS.

## Môi trường và tests

Mọi lệnh Python chạy bằng **conda env `fas`**. Dependencies được khai báo trong
[pyproject.toml](pyproject.toml). Không tạo venv khác hoặc nâng cấp môi trường
đang phục vụ thí nghiệm khác. [Validation mới nhất](results/validation-v14.json):
105 tests passed, gồm actual CUDA checkpoint-resume và pinned FFmpeg helper trên
video tổng hợp.

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
Explicit continuation dùng `plan --previous-plan ...` với budget mới; reuse
caches/retain failures theo pinned predecessor receipts, không overwrite attempt
cũ. Watchdog cộng cả hai roots vào cùng output cap. Chi tiết và kết quả ở
[tài liệu 006](docs/006-explicit-source-continuation.md).

### Bounded adapter pilot

[Pilot runner](scripts/adapter_pilot.py) khóa hai variants pretrained spatial
adapter / temporal adapter, hai epochs, cùng source data/windows/noise schedule;
train bằng private source-only config, không sửa default execution gates.
Epoch-boundary resume và separate-process EMA replay được kiểm tra trên
checkpoints thực tế. Watchdog dùng shared 45-minute / 8-GiB conservative
device-GPU / 2-GiB new-output cap. Xem [protocol và kết quả](docs/007-matched-source-adapter-pilot.md).

### Dedicated source-validation diagnostic

[Diagnostic runner](scripts/source_diagnostic.py) có `plan`, `run`, `report`;
chỉ dùng sau explicit source-validation attack approval với shared finite caps.
Giữ default CLI guards; archive extraction có opt-in chỉ cho source-validation
attacks, không cho fitting/calibration/target attacks. Models/checkpoints/noise
levels/score direction được khóa trước decoding; không training, threshold
fitting hoặc model selection. Export aggregate AUROC, paired group-bootstrap
CIs, all-original coverage và missing-score bounds; individual profiles/caches
giữ private. Xem [protocol, lỗi ban đầu và kết quả](docs/008-source-validation-spoof-diagnostic.md).

### Private crop review before further scoring

[Preview runner](scripts/crop_preview.py) tạo HTML tự chứa ảnh để so original,
legacy crop và CelebA-HQ-style square crop, trên seeded source-validation sample.
Worker chỉ CPU, không DDPM scoring/training hoặc calibration/target pixels.
Crop mới là geometry/resampling ablation, không claim bit-exact preprocessing
của checkpoint hoặc cải thiện PAD. HTML và ảnh giữ ngoài Git. **Người dùng đã
reject việc bù context bằng reflection**; preview này chỉ còn là lịch sử, server
đã dừng và chưa chạy ablation. Xem [protocol](docs/009-celebahq-geometry-visual-review.md).

### Real-pixel-only crop eligibility

[Metadata auditor](scripts/crop_fit_audit.py) kiểm tra nguyên square geometry
trên native frame dimensions, không padding/shift/shrink, decode ảnh hoặc chạy
model. Đã audit 618 caches / 651 source originals: MSU phù hợp nhất nhưng
không dataset nào đạt 100% toàn bộ denominator. MSU có 18/18 fitting bonafides,
6/6 validation bonafides và 16/18 validation attacks đạt; một attack không đủ
context, một attack có decode failure cũ. Việc dùng subset cần chốt protocol
riêng, không tự đổi validation window để cứu mẫu. Xem
[coverage và selection-bias caveats](docs/010-real-pixel-crop-dataset-audit.md).

Đã chốt eligible MSU population bằng metadata và chạy preview CPU được duyệt:
**6/6 validation videos, 18/18 frames**, zero padding, không đổi middle window.
Strict renderer kiểm tra cả square bounds lẫn actual bilinear support trước
khi sample RGB; historical reflected renderer không được dùng cho policy mới.
25 targeted crop tests passed ở bước preview. HTML/ảnh giữ private, **đã được
duyệt bằng mắt**; server preview đã dừng. Xem
[protocol và aggregate](docs/011-msu-real-pixel-crop-review.md).

Frozen-DDPM A/B được duyệt riêng đã hoàn tất trên cùng 22 validation clips:
AUROC **0,687500 legacy / 0,697917 real-pixel square**, delta **+0,010417**,
paired subject-bootstrap 95% CI **[0,000000; +0,083333]**, chỉ 3 subject groups.
Bonafide residual giảm 14,41% nhưng attack residual cũng giảm 10,70%;
**chưa chứng minh separation cải thiện chắc chắn**. 33 targeted crop tests
pass; không train/fit threshold hoặc decode fitting/calibration/target.
Lỗi preflight list/tuple lượt đầu và explicit approval lượt mới được giữ
trong [report 012](docs/012-msu-frozen-crop-ablation.md). Không có lượt chạy tiếp.

```bash
conda run --no-capture-output -n fas python scripts/crop_preview.py plan \
  --diagnostic-root /path/to/private/source-diagnostic-v2 \
  --real-pixel-audit-root /path/to/private/crop-fit-audit-v1 \
  --private-root /path/to/private/new-msu-real-pixel-preview

CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n fas python scripts/crop_preview.py run \
  --plan /path/to/private/new-msu-real-pixel-preview/plan.json
```

Mỗi actual attempt vẫn cần approval/budget riêng; các lệnh không cấp quyền
scoring/training. Audit v1 và original population được pin theo hashes.

### MAT face-inpainting pilot

Đã tích hợp riêng MAT CelebA-HQ-512: **6 videos, 18 frames, 72 reconstructions**
cho mắt trái/mắt phải/mũi/miệng, crop trực tiếp native RGB ở 512, không padding
hay phóng cache 256. Đúng RGB normalization và `1=known, 0=hole`; chỉ đo lỗi
trong hole. Actual synthetic GPU smoke cho replay/no-leakage error bằng 0.
10 targeted MAT tests pass; không cài/nâng cấp package trong `fas`.

**Mirror safetensors bên thứ ba được người dùng duyệt; chưa xác minh weights
khớp bản tác giả.** Link chính thức trả 404. Giữ failed synthetic attempt và
explicit continuation evidence. HTML/ảnh giữ private; **crop và mask đã được
người dùng duyệt bằng mắt**. Preview không chạy AUROC, train hay threshold fitting. Xem
[report, protocol và provenance limits](docs/013-mat-native512-inpainting-preview.md).

Lượt diagnostic tiếp theo có approval/budget riêng: cùng 22 MSU validation
originals, cửa sổ giữa T16, bốn mask độc lập, 1.408 reconstructions. Điểm chính
là MSE trong hole, gộp đều bốn vùng và mười sáu frames; higher=attack.
Không chọn mask tốt nhất/đảo dấu theo kết quả, train hay fit threshold. Xem
[protocol cố định và đối chiếu DDPM](docs/014-msu-mat-anatomical-diagnostic.md).
Lượt chạy đã hoàn tất **22/22 clips, 352 frames, 1.408 reconstructions**,
không lỗi kỹ thuật. AUROC chính **MAT 0,7500**, so **DDPM 0,6979**;
paired delta **+0,0521**, CI95% **[-0,0833; +0,1042]** chứa 0, chỉ ba nhóm
subject nên **chưa chứng minh MAT tốt hơn hoặc generalize**. Nose riêng có
AUROC 0,2604; giữ nguyên trọng số/dấu, không chọn eye-only theo kết quả.
Postflight độc lập từ float32 hole predictions, sklearn và weighted pairwise
bootstrap đã khớp. Không train/fit threshold; đây vẫn là development evidence.

**Đối chiếu DINO đã hoàn tất** trên đúng 22 originals/T16: DINO-Reg mean AUROC
**0,9792**, plain DINO **0,9826**, fixed two-DINO **0,9826** qua cả ba seed
head cũ loại MSU khỏi fitting/validation/calibration, so MAT **0,7500**.
Giữ preprocessing DINO 518 lịch sử (rectangle-to-square, đã được duyệt và
kiểm tra không padding), không dùng crop MAT thay cho head input cũ.
MAT sửa thứ tự 1–2 cặp DINO sai mỗi seed, nhưng sai 24 cặp DINO xếp đúng;
đây là **pair ranking, không phải spoof rescue/FA reduction hay fusion gain**.
Không chọn seed, train, fit threshold hay fusion; không phải cải thiện
0,927→0,983 trên cùng test cũ. Xem [protocol, CI, full seed results và caveats](docs/015-matched-msu-dino-mat-comparison.md).

MAT runtime dùng pinned official source ngoài Git và optional dependency
`timm==1.0.30` đang có sẵn; không import model này từ scientific CLI mặc định.
Không tự tải model hay compile CUDA extension khi chạy pipeline DDPM.

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
