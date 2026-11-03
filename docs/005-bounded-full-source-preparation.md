# 005 - Full source-bonafide preparation có budget

**Batch đã dừng vì lỗi cache-QA; chưa hoàn tất full-source preparation.**
Không tự chạy lại. [Aggregate evidence](../results/source-preparation-v1.json)
ghi `completed: false`, không có completed prepared-manifest.

## Phạm vi đã được duyệt

Chuẩn bị bonafide fitting/validation của CASIA-FASD, MSU-MFSD và SiW-Mv2 cho
fold target OULU-NPU. Không train diffusion, không forward model trên PAD crops,
không decode attack, calibration hoặc OULU. Mọi Python/tests dùng conda env `fas`.
Default scientific gates vẫn đóng; batch dùng private config riêng, chỉ bật
source decoding.

Budget được duyệt: **7.200 giây, 8 GiB worker-tree RSS, 12 GiB private output,
một CPU worker nối tiếp**. Watchdog lấy mẫu mỗi 0,2 giây, không phải
instantaneous OS quota. RSS không bao gồm parent/controller. Nếu vượt cap thì
dừng owned worker group; không tự retry hoặc thay original.

## Denominator khóa trước tracking

| Domain | Fitting originals | Đủ điều kiện | Validation originals | Đủ điều kiện |
|---|---:|---:|---:|---:|
| CASIA-FASD | 36 | 36 | 12 | 12 |
| MSU-MFSD | 18 | 18 | 6 | 6 |
| SiW-Mv2 | 314 | 307 | 105 | 103 |
| **Tổng** | **368** | **361** | **123** | **121** |

Denominator là **491 originals**, không phải chỉ 482 videos đủ điều kiện.
Chín SiW videos quá ngắn (7 fitting, 2 validation) được giữ là
`insufficient_duration`, không bù bằng video khác.

Metadata-only verification xác nhận cả 482 originals còn lại hỗ trợ sampling
**full interval ở 8 FPS** theo tolerance hiện hành, không chỉ middle window.
Ước lượng trước chạy: 25.792 RGB256 frames, 5.070.913.536 raw RGB bytes;
conservative output estimate 5.383.401.454 bytes. Original lớn nhất cần stage
là 85.208.150 bytes, dưới guard 128 MiB.

Selection plan SHA-256:
`8d9c421e647702468a20c0e340ca9921d026e7e6cc9840027d40c13a47a06e36`.

## Kết quả attempt đầu tiên

| Domain/partition | Originals | Prepared | Quá ngắn | Tracking failure | Interrupted | Chưa thử |
|---|---:|---:|---:|---:|---:|---:|
| CASIA fitting | 36 | 36 | 0 | 0 | 0 | 0 |
| CASIA validation | 12 | 12 | 0 | 0 | 0 | 0 |
| MSU fitting | 18 | 18 | 0 | 0 | 0 | 0 |
| MSU validation | 6 | 6 | 0 | 0 | 0 | 0 |
| SiW fitting | 314 | 210 | 7 | 1 | 1 | 95 |
| SiW validation | 105 | 0 | 2 | 0 | 0 | 103 |
| **Tổng** | **491** | **282** | **9** | **1** | **1** | **198** |

- 282 verified caches: 264 fitting + 18 validation, 16.247 grid frames,
  **846 cache-QA windows**. Đây không phải PAD scores.
- Một SiW original thiếu track endpoint, giữ failure `missing_track_endpoint`;
  không extrapolate hoặc chọn video thay thế.
- Dừng sau **3.679,19 giây (~61,3 phút)**, return code 1. Không chạm resource
  cap: sampled peak worker-tree RSS **1,092 GiB**, peak private output khoảng
  **3,00 GiB**. Supervisor `guard_failure: null`.
- Raw staged videos đã được xóa; archive stderr/native diagnostics giữ private.
  Cache của interrupted original không được tính là prepared hoặc đưa vào
  completed manifest. Không sửa receipts để biến interrupted thành success.
- Không có training, source model scores, attack/target decodes, selection
  replacements, native-hash waivers hoặc automatic retries.

### Nguyên nhân và bản sửa sau khi dừng

Worker dừng tại middle-window QA của một cache uniform-grid có số frames lẻ.
`ClipCache` cũ chọn continuous start rồi nearest-neighbor sampling lại timestamps.
Tại half-grid ties với non-binary timestamp origin, rounding FP64 có thể chọn
cùng một frame hai lần, phát sinh:
`distinct native frames required; no duplication/interpolation`.

Đã reproduce lỗi độc lập bằng **17 synthetic frames, origin 0,1 giây**, trước
khi sửa. [Cache reader](../src/diffusionfas/dataset.py) nay kiểm tra uniform grid
như cũ rồi chọn contiguous integer-index window: middle dùng lower center khi
có hai lựa chọn; addressed random window lấy đều trong các valid start indexes.
Không nới native timestamp tolerance, đổi cache bytes hay duplication checks
trong native sampler.

**Đây là thay đổi sampler policy:** cùng seed có thể chọn window khác version
cũ. Chưa có scientific PAD checkpoint nào cần replay xuyên thay đổi này.
Regression kiểm tra distinct/contiguous frames, middle choice, deterministic
addressed draws, valid-start coverage và short/invalid-duration rejection.

Chỉ chạy synthetic validation sau bản sửa, **không rerun original hoặc cache-QA
của attempt cũ**. `implementation_sha256` trong aggregate mô tả đúng worker
trước bản sửa; validation v6 mô tả code sau bản sửa. Không diễn giải test pass
thành full-source preparation thành công.

## Cơ chế thực thi

[Orchestrator](../src/diffusionfas/source_preparation.py) và
[command script](../scripts/prepare_sources.py):

1. Khóa metadata/source-manifest hashes, exact archive members, receipt hashes,
   detector/helper hashes, implementation hashes và caps trong immutable plan.
2. Với từng original, stage đúng một archive member; verify size/hash rồi mới
   decode. Không extract cả archive hoặc nối toàn bộ MSU split ZIP ra disk.
3. Reuse tracking/crop pipeline hiện có, detector chạy CPU. MSU dùng
   [strict native FFmpeg policy](004-strict-msu-decoder-policy.md); domain khác
   dùng OpenCV. Không fallback decoder hay bỏ native RGB checks.
4. Cache full eligible interval ở 8 FPS, không chỉ một clip 2 giây. Xóa raw staged
   original trong `finally`; caches và diagnostics giữ private.
5. Mỗi cache thành công phải đọc lại qua `ClipCache`: một middle window và hai
   addressed random windows, mỗi window có shape `(16, 3, 256, 256)` và finite
   values. Không chạy denoiser hoặc classifier trong bước QA.
6. Giữ technical failures trên originals đã chọn; full-population manifest
   bảo toàn các originals không được chọn. Không resample để che failure.
7. Supervisor và per-original immutable receipts phân biệt prepared, metadata
   failure, preparation failure, interrupted và not attempted. Không coi
   interrupted batch là hoàn thành.

Private data, original IDs, memberships, per-original hashes và decoder logs
không đưa lên Git. Public report chỉ chứa aggregate evidence và commitments.

## Cách chạy

Các caps dưới đây minh họa đúng budget đã duyệt cho batch này, không cấp quyền
cho batch tiếp theo. Dùng private root mới; không chạy lại để ghi đè receipts.

```bash
conda run --no-capture-output -n fas python scripts/prepare_sources.py plan \
  --metadata /path/to/private/metadata.json \
  --source-manifest /path/to/accepted/source_manifest.json \
  --fas-root /path/to/fas \
  --detector /path/to/private/det_10g.onnx \
  --private-root /path/to/private/new-source-preparation \
  --max-seconds 7200 \
  --max-rss-bytes 8589934592 \
  --max-output-bytes 12884901888

conda run --no-capture-output -n fas python scripts/prepare_sources.py run \
  --plan /path/to/private/new-source-preparation/plan.json

conda run --no-capture-output -n fas python scripts/prepare_sources.py report \
  --plan /path/to/private/new-source-preparation/plan.json \
  --out /path/to/new-aggregate-report.json
```

Sau khi đã khóa plan, không sửa các implementation files trước khi worker
hoàn tất. Worker reject input/code commitments khác plan. Report command kiểm
tra supervisor binding và completed-manifest binding trước khi export.

## Regression evidence

[Validation v5 trước batch](../results/validation-v5.json):
**60 tests passed, 0 skipped**.
[Validation v6 sau bản sửa sampler](../results/validation-v6.json):
**62 tests passed, 0 skipped**,
gồm actual pinned FFmpeg helper trên synthetic FFV1, không dùng PAD videos.
[Năm tests orchestration mới](../tests/test_source_preparation.py) kiểm tra:
full-interval cache QA/raw cleanup; exact source scope và all-original manifest;
interrupted staging; changed commitments/immutable attempts; output budget fail
trước root creation.

Đây là engineering validation, không phải PAD accuracy. Không claim
editor-based type-check vì Pylance MCP chưa kết nối được.

## Sau preparation

**Chưa sẵn sàng train pilot:** SiW validation còn 103 eligible originals chưa
thử, và chưa có full-population completed prepared-manifest. Phần tiếp theo cần
explicit approval/budget mới: hoàn tất 198 originals chưa thử, xử lý interrupted
original dưới sampler đã sửa, revalidate reused caches và khóa full manifest.
Không tự rerun batch, overwrite receipts hoặc retry tracking failure.

Usable fitting/validation counts và measured costs sau khi hoàn tất mới là đầu
vào để duyệt spatial-vs-temporal training pilot. Không tự chuyển sang train.
Strict source BPCER 5% calibration policy chưa đổi; CASIA/MSU vẫn thiếu support
cho policy đã chọn. Chưa có scientific training hoặc target evaluation.
