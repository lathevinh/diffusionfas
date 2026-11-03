# 004 - Strict MSU decoder policy trong CLI/config

## Trạng thái

Đã hoàn thành bước wiring sau [probe 003](003-bounded-checkpoint-gpu-and-source-probes.md).
Không chạy full-source preparation, không train pilot, không decode thêm PAD
videos hoặc OULU. Mọi Python/tests chạy trong conda env `fas`; không cài package.

[Validation v4](../results/validation-v4.json): **55 tests passed, 0 skipped**,
gồm native FFmpeg thật trên video FFV1 tổng hợp. Đây không phải kết quả PAD.
Metadata-only binding cũng đã qua cho đúng 2 MSU originals của fixed probe cũ,
không gọi decoder hoặc thay selection.

## 1. Policy rõ ràng, không fallback

[Default config](../configs/offline-v1.json) bổ sung:

```json
{
  "preprocessing": {
    "decoders": {"MSU-MFSD": "fas_strict_ffmpeg"},
    "native_replay_helper_sha256": "8018390bfa8004d42cd52eae21dab8633a39838f1efc523647de5cd9d823a888"
  }
}
```

- MSU dùng accepted strict FFmpeg replay từ checkout `fas`.
- Các domain không override giữ OpenCV. Config cũ không có section này giữ
  legacy OpenCV behavior; không coi đó là policy đã sửa cho MSU.
- Unknown domain/decoder, typo policy keys, thiếu/sai helper hash đều fail
  explicit; không thử decoder khác, đổi hashes hoặc chọn original thay thế.
- Native receipt bridge hiện hỗ trợ non-OULU domains; không tự đoán layout media
  receipts của OULU.
- **Tất cả scientific execution gates vẫn false, budgets vẫn null.**

Decoder policy nằm trong scientific config hash. Đổi policy là thay đổi
configuration được fingerprint, không âm thầm coi checkpoint/config cũ tương
đương. Không có scientific PAD checkpoint nào được train trong phase này.

## 2. Binding và diagnostics

[Native bridge](../src/diffusionfas/native_replay.py) dùng chung cho
[CLI](../src/diffusionfas/cli.py) và [source probe](../src/diffusionfas/source_probe.py).
Không copy FFmpeg decoder sang implementation thứ hai.

Trước detector/decode, CLI kiểm tra:

1. Source/target decode permission và explicit wall/output budgets.
2. `--source-manifest` phải khớp SHA-256 trong imported manifest provenance.
3. Helper file phải khớp SHA-256 trong decoder policy và được import từ đúng
   checkout đã chỉ định.
4. [Shared receipt reader](../src/diffusionfas/fas_import.py) verify transaction
   hash, original/label join, media-receipt hash và media identity.
5. Receipt phải khớp imported video: media hash, decoded-sequence hash, frame
   count, từng native timestamp và từng RGB hash; accepted dimensions phải cố định.

Mỗi replay tiếp tục kiểm tra original media bytes. Strict FFmpeg helper xác
minh full-video RGB sequence và frame count ở cả hai passes; callback preparation
cũng kiểm tra native order/hash trước khi chọn grid 8 FPS. Không chỉ verify
frames được lấy mẫu.

Private diagnostics nằm trong `cache-root/native-replay/`, prefix là hash của
original key. Mỗi pass có immutable JSON + stderr, lưu return code/verified
frames/failure. Không overwrite hoặc đưa originals/paths/face crops vào Git.

Known decode/hash/count/stderr errors và timeout trở thành `preparation_failure`
trên original đó, không phải spoof prediction. Original và historical failures
được giữ trong manifest; những original/domain/partition không được chọn không
đổi. Unexpected programming/I/O errors vẫn raise, không bị nuốt thành success.

Output-byte accounting bao gồm cả native diagnostics và kiểm tra sau mỗi
original, kể cả original thất bại. CLI budgets vẫn là cooperative checks, không
claim instantaneous OS quotas; full run vẫn cần bounded supervisor/budget riêng.

Cache behavior không đổi: full eligible interval ở 8 FPS, không chỉ cache một
window 2 giây; random training windows và fixed validation windows dùng sau đó.

## 3. Cách dùng sau khi được duyệt budget

Không chạy command dưới đây với default disabled scientific gates. Originals
phải được stage/bind bằng explicit mapping trước; decoder không tự giải nén archive.

```bash
PYTHONPATH=src conda run --no-capture-output -n fas python -m diffusionfas prepare \
  --config /path/to/approved-source-preparation.json \
  --manifest /path/to/private/bound-media.json \
  --partition fitting \
  --cache-root /path/to/private/new-cache \
  --out /path/to/private/new-prepared-manifest.json \
  --fas-root /path/to/fas \
  --preprocessing-sha256 <bound-preprocessing-sha256> \
  --detector /path/to/det_10g.onnx \
  --detector-sha256 <bound-detector-sha256> \
  --source-manifest /path/to/accepted/source_manifest.json
```

Missing/mismatched native inputs fail trước khi detector/media decoding chạy.
Không bật target permission hoặc thay calibration policy để chạy source prepare.

## 4. Regression evidence

[Tests](../tests/test_native_replay.py) kiểm tra:

- CLI chọn native cho MSU, legacy OpenCV cho CASIA, giữ historical failures và
  untouched target originals.
- Full interval synthetic 3 giây tạo 24 cached frames, thay vì cố định 16 frames.
- Actual pinned FAS helper chạy qua CLI, verify đủ 48 native frames mỗi pass
  trên FFV1 tổng hợp; face provider là fixture, không claim detector benchmark mới.
- Corrupt RGB evidence tạo technical failure, không gọi OpenCV fallback.
- Permission gates, thiếu source manifest, sai helper/manifest/receipt/frame
  bindings đều fail closed trước detector/decode.
- Timeout, extra frames, nonzero decoder exit và unexpected errors không biến
  thành success hoặc làm mất original.
- Selection receipt commitment, original bytes, immutable diagnostics và
  diagnostics-output budget được kiểm tra.

Full suite cũng phát hiện flaky spatial-adapter assertion cũ khi so batch shapes
khác nhau: fixed seeds 5/7 lệch FP32 khoảng `1,5e-6` tới `1,7e-6`, đủ làm tolerance
gần zero fail dù không mix frames. [Test](../tests/test_models.py) nay kiểm tra
đúng invariant mạnh hơn: thay duy nhất frame cuối, các frame còn lại phải
**bitwise identical** với cùng batch shape, còn output frame bị thay phải đổi.
Không sửa production model hoặc nới numeric tolerance để ép pass.

```bash
conda run --no-capture-output -n fas python scripts/validate.py \
  --fas-root /path/to/fas \
  --out /path/to/new-validation-receipt.json
```

Không cung cấp FAS root thì portable suite skip riêng actual native-helper
integration; tests khác vẫn chạy. Pylance MCP chưa kết nối được trong môi trường
này, nên không claim editor-based type-check; syntax compilation và runtime
regressions đã thực hiện.

## 5. Bước tiếp theo

Chốt budget full source-bonafide preparation cho fold target OULU: metadata trước
tracking có 361 fitting + 121 validation originals. Sau preparation mới biết
usable counts/cost thực tế để chốt spatial-vs-temporal training pilot.

Chưa thay strict BPCER 5% calibration policy; CASIA/MSU vẫn thiếu support cho
policy đã chọn. Chưa training thật hoặc evaluate target.
