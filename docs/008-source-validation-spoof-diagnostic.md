# 008 - Source-validation spoof-separation diagnostic

## Approval, scope và protocol

Sau [pilot 007](007-matched-source-adapter-pilot.md), người dùng duyệt một
**development diagnostic**, không train dài thêm: tối đa **5.400 giây cho
prepare/scoring/bootstrap**, worker-tree RAM 8 GiB, conservative device-wide
GPU memory 8 GiB, new output 3 GiB; CPU preparation nối tiếp và ba models lần lượt.
Mọi Python/tests dùng conda env `fas`; không automatic retry.

Initial plan SHA-256:
`4445ea6ecad7c5fd29a671cf58c315b56d80a46ebdf657567501e03010aeeae4`.

Plan lượt mới sau explicit reapproval:
`89056dd9d75a0424dd9ead70ad25fa39132db6e2ed2cbf0d220526faf520e966`.
Exact original entries, checkpoints, authority và budgets bằng lượt đầu;
implementation fingerprint thay đổi để bind bản sửa staging.

Không đọc calibration/target pixels hoặc chạy model trên chúng, không fit
normalization/threshold, không đổi checkpoint hoặc chọn model sau khi thấy
attack results. Full input metadata chỉ dùng để bind lineage/isolation.
Default scientific gates và CLI validation-attack guards giữ nguyên.

[Dedicated runner](../scripts/source_diagnostic.py) có opt-in authority riêng:
source-validation attacks được phép, model selection/threshold fitting không
được phép; fixed higher score = attack. Đây là mở rộng có chủ đích cho
development diagnostic, không biến validation thành calibration hoặc test.

## Denominator khóa trước decoding

| Domain | Bonafide originals | Reused usable bonafide | Attack originals | Metadata-eligible attacks |
|---|---:|---:|---:|---:|
| CASIA-FASD | 12 | 12 | 36 | 36 |
| MSU-MFSD | 6 | 6 | 18 | 17 |
| SiW-Mv2 | 105 | 99 | 106 | 98 |
| **Tổng** | **123** | **117** | **160** | **151** |

Giữ đủ **283 originals**, không sample subset sau khi nhìn decoder/detector
outcome. Initial failures: sáu bonafide preprocessing failures đã biết, một
MSU historical decode failure, tám SiW attacks quá ngắn. Không retry/replace.
Tracking failures phát sinh trong attack preparation phải tiếp tục được giữ.

Largest staged original 82.651.142 bytes; conservative new-output estimate
1.919.935.588 bytes. Native receipts/timestamps/RGB hashes được bind trước
decode; MSU vẫn dùng strict FFmpeg, không fallback hoặc waive hashes.

Attack caches dùng full eligible interval ở 8 FPS, cùng preprocessing contract
với bonafide caches. Model scoring dùng fixed middle T16/RGB256 window.
Raw staged original được xóa sau mỗi attempt; caches/diagnostics giữ private.

## Ba models cố định và scoring

1. Frozen pinned pretrained DDPM.
2. Best spatial-adapter EMA từ pilot 007.
3. Best temporal-adapter EMA từ cùng pilot.

Bind cả best/last checkpoint hashes và completed training binding. Không học
adapter mới, không chọn checkpoint bằng attacks. Noise seed, alpha-bars
`[0.9, 0.5, 0.1]`, hai draws/level và backbone frame microbatch 4 giữ như pilot.
Score là **float64 mean của ba residual-profile values; cao hơn = attack**.
Không đảo score nếu AUROC nhỏ hơn 0,5 hoặc chọn noise level sau kết quả.

Reuse 117 bonafide profiles/model từ private pilot evidence: phải khớp exact
original population, domains, finite profile shape và published pilot domain
means. Adapter profiles bind vào saved checkpoints; frozen profiles được pin
SHA riêng trong diagnostic plan. Mỗi model replay ba bonafide originals cố định
(một/domain) trước khi chấp nhận profile reuse, tolerance `rtol=1e-5, atol=1e-6`.
Chỉ attack profiles là new full-cohort model scoring.

## Metrics và uncertainty

- AUROC usable từng domain, attack là positive class, tie xử lý bằng average
  rank. Macro chỉ là trung bình ba domain AUROCs, không pooled-camera ranking.
- Coverage/failure counts luôn dùng original denominator cho từng class.
  Technical failure có `score: null`, không được tự biến thành spoof prediction.
- Report thêm lower/upper all-original AUROC sensitivity bounds: giữ observed
  scored-pair wins, cho các missing-score pairs lần lượt worst/best case.
  Đây không phải deployed fail-closed operating metric.
- Paired cluster bootstrap **2.000 replicates/domain**: cùng sampled groups cho
  ba models, giữ cả hai class và failed originals trong mỗi declared group;
  báo model CIs và paired AUROC-difference CIs.
- Replicates thiếu usable class được đếm rõ; CI unavailable nếu valid fraction
  dưới 95%, không fabricate AUROC hoặc âm thầm bỏ invalid replicates.
- CASIA chỉ có **4 validation subject groups**, MSU **3**: support rất hạn chế.
  SiW dùng declared original/group; subject identity chưa được xác minh nên
  không gọi đây là subject-level uncertainty.

Attack-family mapping chưa được xác minh cho cohort này. Không đoán
print/replay/subtype từ filenames hoặc archive layout.

## Resource accounting và validation

Watchdog dùng chung wall/RSS/GPU/output caps cho toàn job; không cấp thêm budget
cho từng model. Device-wide GPU telemetry bao gồm ứng dụng khác; nếu vượt cap,
chỉ dừng owned worker group. Own PyTorch allocator tối đa 7 GiB, chừa overhead.
Nominal watchdog interval 0,2 giây cộng telemetry-call time, không phải
instantaneous OS quota.

[Validation v11](../results/validation-v11.json): **84 tests passed, 0 skipped**,
gồm actual CUDA checkpoint-resume và native FFmpeg synthetic integration.
New tests kiểm tra exact validation scope, opt-in authority, unchanged CLI
guards, reused-profile provenance, source receipt label binding, fixed AUROC
direction/ties, all-original failure bounds, paired cluster deltas và explicit
unavailable metrics. Không dùng PAD attacks trong unit tests.

### Lượt đầu dừng và root-cause fix

[Aggregate v1](../results/source-diagnostic-v1.json) giữ nguyên failed attempt:
3,073 giây, returncode 1, không resource-guard violation, **chưa có AUROC**.
Worker ghi hai reused bonafide receipts rồi chạm attack đầu tiên; archive
stager cũ từ bonafide-only probe từ chối label attack trước khi mở archive.
Attack đó là interrupted, 280 originals còn unattempted; không attack decoding,
new crop hoặc model forward nào đã diễn ra.

Root-cause fix thêm explicit `allow_validation_attacks` opt-in vào shared
archive stager, chỉ chấp nhận non-target source **validation** attacks.
Default vẫn từ chối attacks; flag không cho fitting/calibration/evaluation
attacks hoặc OULU. Diagnostic worker chỉ truyền flag sau checked authority.
Không relabel attack thành bonafide hoặc làm yếu byte/hash/size/deadline guards.

[Validation v12](../results/validation-v12.json): **86 tests passed, 0 skipped**.
Hai tests mới bao phủ closed-by-default staging/split/domain matrix và actual
diagnostic worker extract một synthetic attack ZIP tới preparation boundary,
verify bytes, cleanup raw, propagate unexpected errors và reject second run.
V11 đã không bắt được integration gap này; không gọi unit-suite success là
real-data diagnostic completion.

Người dùng duyệt một lượt mới riêng: 90 phút / RAM 8 GiB / conservative device
GPU 8 GiB / tối đa 3 GiB **new output của lượt mới**, giữ riêng khoảng 5,6 MiB
evidence lượt đầu. Lượt cũ không overwrite; không automatic retry.

## Kết quả lượt được duyệt mới: hoàn tất

[Aggregate v2](../results/source-diagnostic-v2.json) có `completed: true`.
Wall **2.845,744 giây = 47,43 phút**, sampled worker-tree peak RAM
**2,524 GiB**, sampled device-wide peak GPU **5,535 GiB**, sampled peak output
**1,460 GiB**, final private output khoảng **1,440 GiB**. Không cap violation.

### Coverage và failures

| Domain | Usable bonafide / originals | Usable attacks / originals |
|---|---:|---:|
| CASIA-FASD | 12 / 12 | 36 / 36 |
| MSU-MFSD | 6 / 6 | 17 / 18 |
| SiW-Mv2 | 99 / 105 | 89 / 106 |
| **Tổng** | **117 / 123** | **142 / 160** |

Đủ **283 final preparation receipts**, không còn interrupted/unattempted;
259 originals usable, 24 technical failures. Cả 151 eligible attacks được
attempt đúng plan: 142 prepared, chín tracking failures mới, không replacement.
117 bonafide caches không decode lại.

- MSU: một attack historical decode failure đã biết.
- SiW bonafide: hai short và bốn missing-track-endpoint failures đã biết.
- SiW attacks: tám short đã biết; sáu missing endpoints, hai ambiguous track
  jumps, một track gap vượt giới hạn phát sinh mới.

### Fixed-direction usable-only AUROC

Các số bên dưới là AUROC scale `[0,1]`, **higher residual = attack**.

| Domain | Frozen | Spatial adapter | Temporal adapter | Temporal 95% cluster-bootstrap CI |
|---|---:|---:|---:|---|
| CASIA-FASD | 0.381944 | 0.381944 | 0.381944 | [0.289352, 0.444444] |
| MSU-MFSD | 0.676471 | 0.676471 | 0.676471 | [0.600000, 0.750000] |
| SiW-Mv2 | 0.500624 | 0.500738 | 0.501192 | [0.419142, 0.583582] |
| **Domain macro point** | **0.519680** | **0.519718** | **0.519869** | Không ước lượng macro CI |

Mỗi domain có đủ **2.000/2.000 valid bootstrap replicates**, không one-class
replicate. CASIA/MSU delta giữa ba models bằng zero cho mọi bootstrap draw:
bonafide MSE thay đổi nhưng usable ranking không cải thiện.

Trên SiW, temporal-minus-frozen AUROC chỉ **+0.000567**,
95% paired CI **[-0.000232, +0.001718]**; temporal-minus-spatial
**+0.000454**, CI **[-0.000248, +0.001494]**. Chưa có bằng chứng temporal
adapter tốt hơn matched spatial hoặc frozen theo separation metric này.
Domain-macro temporal improvement chỉ **0,0189 percentage point** so frozen.
CI không được dùng để chọn model hay tạo claim subject generalization.

### All-original sensitivity bounds

Không coi usable-only point estimate là full-population PAD performance.
Với temporal model, possible AUROC interval nếu các missing scores lần lượt
worst/best case:

| Domain | All-original missing-score bounds |
|---|---|
| CASIA-FASD | [0.381944, 0.381944] |
| MSU-MFSD | [0.638889, 0.694444] |
| SiW-Mv2 | [0.396765, 0.605121] |

Bounds là missing-data sensitivity, **không phải confidence intervals**.
Failure bounds/coverage và CIs cho cả ba models có đầy đủ trong aggregate.

### Postflight đã kiểm chứng

- Exact 283-original order/identity ở final manifest, receipts và cả ba
  prediction populations; failed originals luôn có null score/profile và reason.
- SHA-256 toàn bộ **142 new attack caches + 117 reused bonafide caches** đúng;
  không staged raw còn sót, chỉ retained extractor stderr diagnostics.
- Base prepared manifest, pilot lineage, locked best/last checkpoints, assets
  và implementation fingerprints vẫn khớp plan sau execution.
- 117 copied bonafide profiles/model khớp pinned pilot evidence từng vector;
  ba replay originals/model có maximum absolute error **0,0**.
- Mọi usable score đúng float64 mean ba levels; không đảo chiều/fit threshold.
- AUROCs được tính lại độc lập bằng explicit attack-bonafide pair comparisons
  và đối chiếu `sklearn.metrics.roc_auc_score`; failure bounds cũng được kiểm lại.
- Tất cả model và pairwise bootstrap CIs được tính lại bằng group multiplicities
  và weighted pair-comparison matrices, thay vì gọi lại bootstrap implementation.
- Public JSON được kiểm tra không có private data paths hoặc individual
  identities/profiles. Cả 160 attack families vẫn unknown, không subtype claims.
- Training/calibration/target pixel access hoặc model forwards ngoài scope:
  zero; không automatic retry, score flip hoặc checkpoint selection.

## Quyết định và bước đề xuất

**Chưa nên train dài hơn hoặc mở target evaluation với recipe hiện tại.**
Macro AUROC chỉ khoảng 0,52; CASIA còn ngược hướng dự đoán ban đầu, SiW gần
random. Không đảo score CASIA để làm đẹp kết quả. Denoising MSE giảm ở pilot
không chuyển thành spoof separation có ý nghĩa.

Kết quả này không phủ định mọi diffusion-based PAD: chỉ là adverse evidence cho
pretrained DDPM + tiny external adapter + raw mean residual recipe đang khóa.
Chưa có matched transformer baseline nên không kết luận diffusion thua
transformer từ số liệu hiện tại.

Bước tiếp theo hợp lý là một **bounded frozen DINO one-class baseline** dùng
cùng original population/crops, fitting bonafide-only và fixed source diagnostic;
đồng thời kiểm tra nuisance strata chỉ khi metadata đã được xác minh. Cần duyệt
asset provenance, protocol và budget riêng trước khi chạy. Nếu đổi diffusion
score/noise weighting, phải predeclare và coi đây là development iteration;
không tune bằng target hoặc trình bày cùng validation cohort như unbiased test.
Không tự triển khai bước tiếp theo trong lượt này.

## Giới hạn diễn giải

Bonafide validation đã được dùng chọn checkpoint ở pilot. Kết quả này là
**development diagnostic conditional on fixed models**, không phải unbiased
test performance hoặc cross-domain generalization proof. Không có deployed
threshold, APCER/BPCER/ACER, target AUC hoặc calibration-policy thay đổi.
Strict source BPCER 5% blocker vẫn nguyên trạng.

Only aggregate evidence được publish; IDs, memberships, profiles, crops,
checkpoints và decoder logs giữ private.
