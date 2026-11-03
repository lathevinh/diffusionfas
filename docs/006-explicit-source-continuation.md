# 006 - Tiếp tục source preparation sau explicit approval

**Đã hoàn tất full-source preparation, chưa train model.**
[Aggregate v2](../results/source-preparation-v2.json) ghi `completed: true`;
full-population prepared-manifest và toàn bộ caches giữ private.

## Kết quả cuối cùng

| Domain/partition | Originals | Prepared | Quá ngắn | Tracking failure |
|---|---:|---:|---:|---:|
| CASIA fitting | 36 | 36 | 0 | 0 |
| CASIA validation | 12 | 12 | 0 | 0 |
| MSU fitting | 18 | 18 | 0 | 0 |
| MSU validation | 6 | 6 | 0 | 0 |
| SiW fitting | 314 | 305 | 7 | 2 |
| SiW validation | 105 | 99 | 2 | 4 |
| **Tổng** | **491** | **476** | **9** | **6** |

- Usable pool: **359 fitting + 117 validation originals**.
- Caches chứa **25.552 RGB256 grid frames ở 8 FPS**; QA đủ **1.428 windows**
  2 giây, gồm cả 282 caches reuse. Không coi các windows là independent videos.
- Chuẩn bị mới 199 originals theo đúng selection: 194 thành công, 5 thiếu track
  endpoint. Tracking failure cũ được giữ, tổng 6; không extrapolate hoặc retry.
- Không còn interrupted/unattempted originals. `completed: true` nghĩa là toàn
  bộ 491 originals đã được accounting, **không** có nghĩa 491 videos usable.
- Continuation chạy **2.207,94 giây (~36,8 phút)**, sampled peak worker-tree RSS
  **1,477 GiB**, combined private output **~4,83 GiB**. Không chạm cap.
  Hai attempts cộng khoảng 98,1 phút worker wall time.
- Không còn raw staged videos ở cả hai roots; private diagnostics được giữ.
  Cache orphan của interrupted attempt trước không thuộc final usable pool.

Postflight đã đọc full manifest, verify hashes của **cả 476 referenced caches**,
kiểm tra đủ 491 result receipts, đúng 199 stage/prepare calls trong continuation,
không thay bất kỳ unselected original nào, và giữ đúng failure denominator.
Final manifest SHA-256:
`c2aa3ead64981a863fce6a78339a929c6d037f43609a9ff6a36988a183b84a7d`.

Không có training, source model scores, attack/target decodes, selection
replacements, native RGB waivers hoặc automatic retries. Interrupted original
được xử lý lại chỉ sau explicit approval cho continuation.

## Approval và scope

Sau [attempt 005 bị dừng](005-bounded-full-source-preparation.md), người dùng
đã duyệt rõ lượt tiếp theo: **tối đa 3.600 giây chạy, 8 GiB worker-tree RSS,
tổng output hai attempts không quá 12 GiB, một CPU worker**. Không coi approval
batch trước là quyền tự retry. Mọi Python/tests vẫn dùng conda env `fas`.

Selection vẫn giữ nguyên **491 source bonafide fitting/validation originals**:

- Revalidate và reuse 282 prepared caches, không stage/decode lại originals.
- Retain một tracking failure cũ, không retry hoặc thay video.
- Retain chín short-video metadata failures.
- Chỉ stage/prepare **199 SiW originals**: 198 chưa thử và một interrupted.
  Cache orphan của interrupted original trong attempt cũ không được reuse.
- Không đọc attack, calibration hoặc OULU, không forward diffusion, không train,
  không fit normalization/threshold.

Continuation plan SHA-256:
`879f5d89710108d5e9b9ad9efe4b3b16225049199e63623c2829aad385765104`.

## Binding và resource accounting

[Orchestrator](../src/diffusionfas/source_preparation.py) hỗ trợ một predecessor
đã có supervisor receipt. New attempt dùng **private root mới, disjoint**;
không overwrite plan, records, caches hoặc logs của attempt trước.

Plan bind predecessor plan/supervisor/per-original receipt hashes, exact
population, imported evidence và detector/preprocessing/decoder policy.
Prepared-cache bytes phải khớp old receipt; cache path phải đúng địa chỉ thuộc
old attempt. Worker reject predecessor receipt thay đổi trước detector/staging.
Implementation hashes được khóa riêng cho code mới, bao gồm sampler đã sửa.

Mỗi reused cache tiếp tục qua ba cache-QA windows bằng corrected integer-grid
sampler. Final records đánh dấu reuse, còn cached files giữ nguyên ở private
root cũ. Full manifest bind cả current plan và predecessor plan để giữ lineage.
Hai roots đều phải được giữ để các cache references còn hợp lệ.

Watchdog cộng output bytes của **cả hai roots**, không cấp thêm 12 GiB cho root
mới. Worker cũng trừ combined usage khi cấp per-original output budget.
Conservative combined preflight estimate: **8.602.905.614 bytes**, dưới cap.
Sample interval 0,2 giây; đây không phải instantaneous OS quota. RSS đo
worker/descendants, không gồm controller.

## Command

Chỉ dùng sau explicit approval mới. Giữ nguyên original metadata và scientific
default config; không dùng partial manifest để tự đổi source population.

```bash
conda run --no-capture-output -n fas python scripts/prepare_sources.py plan \
  --metadata /path/to/private/original-metadata.json \
  --source-manifest /path/to/accepted/source_manifest.json \
  --fas-root /path/to/fas \
  --detector /path/to/private/det_10g.onnx \
  --private-root /path/to/private/new-continuation \
  --previous-plan /path/to/private/previous-attempt/plan.json \
  --max-seconds 3600 \
  --max-rss-bytes 8589934592 \
  --max-output-bytes 12884901888

conda run --no-capture-output -n fas python scripts/prepare_sources.py run \
  --plan /path/to/private/new-continuation/plan.json

conda run --no-capture-output -n fas python scripts/prepare_sources.py report \
  --plan /path/to/private/new-continuation/plan.json \
  --out /path/to/new-aggregate-report.json
```

Chưa hỗ trợ chuỗi nhiều predecessor attempts. Không tự chuyển một failed
continuation thành lần retry tiếp theo.

## Validation

[Validation v7](../results/validation-v7.json): **67 tests passed, 0 skipped**,
gồm actual pinned native FFmpeg helper trên synthetic media.
Năm regressions bổ sung kiểm tra:

1. Reuse cache, revalidate windows mà không stage hoặc decode lại.
2. Interrupted original chỉ được prepare trong explicit new attempt.
3. Retain tracking failure, không retry.
4. Changed predecessor receipt fail trước detector/staging.
5. Watchdog phát hiện combined disk cap dù riêng new root còn nhỏ.

Không có scientific PAD training/evaluation trong validation này.

## Bước tiếp theo

Có thể chốt budget riêng cho **matched spatial-adapter vs temporal-adapter
training pilot** trên 359 fitting originals, validation source-only trên 117
originals; phải khóa cùng backbone, noise addresses và stopping rule.
Chưa tự bật source-training permission hoặc chạy pilot.

Strict BPCER 5% calibration policy chưa đổi; CASIA/MSU vẫn thiếu support cho
policy đã chọn. Preprocessing thành công không giải quyết calibration blocker
hoặc chứng minh cross-domain PAD accuracy. Không dùng OULU để sửa policy.
