# 001 - Kế hoạch diffusion bonafide-only cho short-clip FAS

Ngày: 2026-11-03.

**Trạng thái: đề xuất để phản biện, chưa triển khai hoặc thực nghiệm.**

Cập nhật sau quyết định triển khai: xem
[002 - implementation và preflight](002-offline-implementation-and-preflight.md).
001 được giữ làm proposal lịch sử; 002 mô tả code thực tế, hướng pretrained được
chọn và các data/calibration blockers, không hàm ý scientific training đã chạy.

Tài liệu này chuyển phương án đã trao đổi thành kế hoạch có thể triển khai.
Các hyperparameter là lựa chọn nghiên cứu ban đầu, không phải recipe đã chứng
minh hiệu quả. Việc đưa tài liệu lên repository không cho phép tự động decode
dữ liệu, cài dependency, tải trọng số, training hoặc đánh giá target.

## 1. Mục tiêu và giả thuyết

Hai câu hỏi chính:

1. Diffusion chỉ học bonafide có tạo được anomaly score hữu ích cho face
   presentation attack detection (PAD), đặc biệt trên domain mới không?
2. Temporal modeling có cải thiện vượt việc chấm nhiều frame độc lập rồi
   tổng hợp score không?

Giả thuyết cần kiểm chứng: phân phối không gian-thời gian của clip bonafide
có thông tin phân biệt bổ sung so với phân phối từng ảnh.

Không giả định rằng:

- Model chỉ học bonafide sẽ luôn tái tạo spoof kém.
- Reconstruction/noise error là xác suất spoof.
- Chuyển động, chớp mắt hoặc rPPG đủ chứng minh một clip là live.
- Diffusion tốt hơn transformer: diffusion là cách mô hình hóa, transformer
  là kiến trúc. Cần đối chiếu cùng chế độ one-class.

Rủi ro trung tâm: model có thể nhạy với camera, ánh sáng, crop hoặc compression
hơn với spoof. Replay chất lượng cao có thể vẫn nằm gần phân phối bonafide.

## 2. Bằng chứng hiện có và giới hạn

Nguồn dữ liệu và pipeline hiện tại nằm ở repository
[lathevinh/fas](https://github.com/lathevinh/fas).
Đây là nguồn tham chiếu, không phải code/data đã được chuyển sang repository này.

### 2.1. Dữ liệu

Đã quan sát archive gốc của OULU-NPU, CASIA-FASD, MSU-MFSD và SiW-Mv2.
SiW có video MOV/MP4/AVI, không chỉ có ảnh.

Theo [dataset inventory hiện có](https://github.com/lathevinh/fas/blob/main/manifests/dataset_summary.csv):

| Dataset | Bonafide toàn inventory | Attack toàn inventory |
|---|---:|---:|
| OULU-NPU | 990 | 3.960 |
| CASIA-FASD | 150 | 450 |
| MSU-MFSD | 70 | 210 |
| SiW-Mv2 local intersection | 785 | 895 |

Đây không phải số video được phép train hoặc số clip hợp lệ. Cần lọc theo
official partition, source-use exclusions, fitting/validation/calibration
và khả năng tạo clip đủ thời lượng.

SiW là local Protocol-I intersection, không phải toàn bộ release. Subject
identity của SiW chưa biết; không được tuyên bố participant-disjoint.

Năm source-use exclusions của SiW đã được thông qua trong
[tài liệu 293](https://github.com/lathevinh/fas/blob/main/docs/293-siwmv2-adopted-source-use-and-saved-evidence-closure.md).
Phải giữ exclusions khi xây manifest mới; không phục hồi video bị loại chỉ
vì đổi từ ảnh sang clip.

### 2.2. Pipeline và tài nguyên

- Pipeline cũ chủ yếu dùng một middle frame; cache ảnh/feature cũ không đủ
  cho temporal modeling.
- Có SCRFD và quy tắc crop trong
  [preprocessing v3](https://github.com/lathevinh/fas/blob/main/configs/preprocessing_v3.yaml).
- Có timestamp/decode evidence và các adapter cần đánh giá khả năng tái sử dụng,
  không nên viết lại decoder riêng nếu không cần thiết.
- Probe lịch sử trong
  [tài liệu 329](https://github.com/lathevinh/fas/blob/main/docs/329-implemented-controlled-dino-adaptation-package.md)
  từng dùng RTX 4080 SUPER. GPU hiện khả dụng, VRAM trống và throughput diffusion
  chưa được xác minh. Lệnh `nvidia-smi` không có trên PATH trong lần kiểm tra này;
  điều đó không chứng minh máy không có GPU.
- Diffusers chưa nằm trong dependency lock đã kiểm tra của pipeline cũ.

Không ước lượng thời gian training từ throughput DINO trước đây.

### 2.3. Điều cần port trước implementation

Repository mới cần một export có version của protocol, source-use policy và
manifest bindings. Không chỉ dựa vào link đến nhánh `main` có thể thay đổi.
Trước execution phải pin commit/hash nguồn, split và cấu hình preprocessing.

Không đưa raw video, ảnh mặt, private IDs, absolute data paths, row-level
membership hoặc individual predictions lên Git. Chỉ public aggregate counts,
schema, config và hash phù hợp.

## 3. Protocol và quyền sử dụng dữ liệu

### 3.1. Bốn outer folds

| Fold | Source datasets | Held-out target |
|---|---|---|
| 1 | CASIA + MSU + SiW | OULU |
| 2 | OULU + MSU + SiW | CASIA |
| 3 | OULU + CASIA + SiW | MSU |
| 4 | OULU + CASIA + MSU | SiW |

Kế thừa partition của
[protocol 319](https://github.com/lathevinh/fas/blob/main/docs/paper319/supplementary-protocol.json),
không random split lại để tăng support hoặc làm kết quả đẹp hơn:

- Source fitting: chỉ bonafide thuộc fitting partition.
- Source validation: bonafide validation; không dùng target.
- Source calibration: partition riêng, không dùng để cập nhật model weights.
- Source official test không dùng cho fitting, selection hoặc calibration.
- Target official test chỉ scoring sau khi khóa toàn bộ recipe.

Tất cả window từ cùng video phải cùng partition. Giữ subject/content-component
grouping khi có. Không coi các frame hay window là đơn vị độc lập.

Nếu clip eligibility làm một partition thiếu support, báo blocked trước khi
training; không chuyển video giữa các partition.

### 3.2. Hai chế độ calibration phải báo riêng

**A. Strict one-class:** fitting, checkpoint selection, score construction và
threshold chỉ dùng bonafide nguồn.

**B. Bonafide-only training, attack-aware calibration:** giữ nguyên model và
score của A; chỉ dùng attacks trong source calibration để đặt threshold.

B không được dùng source attacks để chọn kiến trúc, timestep, augmentation,
checkpoint hoặc score rồi vẫn gọi là strict one-class.

## 4. Data preparation

### 4.1. Inventory trước bulk extraction

Xuất bảng aggregate theo dataset/partition/class:

- Số video eligible theo source-use policy.
- Duration và khoảng thời gian hợp lệ theo official protocol.
- Số video đủ 1, 2 và 3 giây.
- Decode failures và các giới hạn timestamp/codec.
- Số subject/group và support bonafide fitting/validation/calibration.

Không suy ra training support từ tổng số frame.

### 4.2. Manifest clip

Private records tối thiểu:

```text
dataset, original_video_key, subject_or_group_key
official_split, experiment_partition, binary_label
valid_time_interval
clip_start, clip_duration
selected_frame_timestamps
decode_status, tracking_status, failure_reason
source_video_hash, preprocessing_hash
```

Train windows có thể được xác định theo seed/epoch/video thay vì materialize
mọi window; phải replay được sampling.

### 4.3. Sampling

Primary input: **2 giây, 8 frame/giây, 16 frame**.

- Chọn frame gần các timestamp định trước; không tạo frame nội suy.
- Training: chọn source đều nhau, chọn video trong source, rồi random window
  trong khoảng hợp lệ.
- Video dài không được tăng trọng số chỉ vì có nhiều window.
- Validation/test: một window giữa khoảng hợp lệ, khóa trước detection/scoring.
- Không tìm window đẹp nhất theo detection, reconstruction hoặc anomaly score.

Quy tắc chọn frame gần nhất, xử lý ties và sai lệch timestamp tối đa phải được
định nghĩa và test trước extraction. Nếu không tạo được 16 frame phân biệt ở
các mốc chấp nhận được, ghi failure; không lặp frame ngầm.

Video không đủ 2 giây: `insufficient_duration`. Không loop, pad bằng frame lặp
hoặc tự chuyển sang clip ngắn. Nếu tỷ lệ này quá cao, phản biện và sửa protocol
trước scientific fitting.

### 4.4. Detect, track và crop

Kế thừa SCRFD hiện có; thêm tracking nhất quán trong clip:

1. Detect trên các frame đã chọn.
2. Liên kết cùng một mặt qua clip.
3. Làm mượt box/landmark để giảm crop jitter.
4. Similarity alignment, không warp phi tuyến khuôn mặt.
5. Context khoảng 1,3 lần, giữ hướng tiếp cận của pipeline cũ.

Proposal ban đầu cho tracking:

- Nội suy tối đa hai sampled frame mất detection liên tiếp, cần anchor hai phía
  và track nhất quán.
- Không ngoại suy đầu/cuối clip.
- Track không đủ điều kiện là technical failure, không đổi window để cứu mẫu.
- Lưu landmarks và transforms để phân biệt motion mặt với crop motion.

Định nghĩa association và mức smoothing cần khóa sau source engineering checks.
Không chọn bằng target PAD performance.

### 4.5. Cache

- Canonical crop: RGB 256x256, uint8/lossless.
- Training primary: resize về 128x128.
- Không beauty filter, thêm denoising hoặc normalize màu mạnh.
- Giữ backbone normalization ngoài cache; DDPM dùng một mapping input xác định
  sang [-1, 1].
- Cache theo video/frame để window dùng chung dữ liệu, không lưu trùng cả clip.
- Không ghi đè crop/cache cũ.

Đo storage và decode/crop throughput trên một tập source nhỏ trước bulk work.
128x128 là feasibility configuration, không bảo đảm còn đủ high-frequency cues.

### 4.6. Leakage/content checks

Kế thừa exclusions và evidence byte/full-decoded-sequence đã có. Kiểm tra thêm
exact selected-clip content crossings giữa các partition.

Một frame trùng không tự chứng minh hai clip/capture giống nhau. Các nghi vấn
near-duplicate phải được ghi nhận và xử lý có version, không tự động đổi split.
Không tuyên bố capture independence chỉ từ exact-hash checks.

## 5. Model

### 5.1. Lựa chọn chính

**Pixel-space DDPM dự đoán epsilon, U-Net nhỏ, random initialization.**

Tham chiếu implementation:

- [Diffusers UNet2DModel](https://huggingface.co/docs/diffusers/api/models/unet2d).
- [Diffusers DDPMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddpm).

| Thành phần | Cấu hình đề xuất |
|---|---|
| Input/output | RGB / noise RGB |
| Resolution | 128x128 |
| U-Net channels | 64, 128, 192, 256 |
| Residual layers/block | 2 |
| Spatial attention | Chỉ ở mức resolution thấp |
| Training timesteps | 1.000 |
| Schedule | Cosine |
| Prediction | Epsilon |
| External pretrained weights | Không |

Không bắt đầu bằng Stable Diffusion/latent video diffusion: VAE có thể làm mất
cue nhỏ, external pretraining làm claim phức tạp và chi phí khó kiểm soát.
Không tuyên bố đây là exact reproduction của DiffusionAD hoặc DiffFAS.

Trước implementation phải pin phiên bản dependency tương thích và topology
đầy đủ. Dùng môi trường riêng; không thay dependency của scientific run khác.

### 5.2. D2: spatial diffusion

Áp dụng cùng U-Net độc lập cho các frame:

```text
[B, T, C, H, W] -> [B*T, C, H, W] -> noise prediction
```

Một checkpoint cung cấp cả single-middle-frame score và multi-frame score.

### 5.3. DT: temporal diffusion

Giữ spatial U-Net; thêm residual temporal convolution kernel 3 tại hai mức
feature thấp và bottleneck.

- Temporal branch dùng zero-init output projection để khởi đầu gần identity.
- Không temporal downsampling.
- Không full-resolution temporal attention.
- Không condition bằng camera/domain label.
- D2 và DT dùng cùng spatial initialization và addressed sampling/augmentation.
- Train riêng từ đầu, không cho DT thêm warm-start budget.

Khóa chính xác vị trí temporal blocks sau engineering probe. Cùng số optimizer
steps không có nghĩa cùng FLOPs; phải báo params, compute và latency riêng.

## 6. Training

### 6.1. Objective

Với clip X, timestep t và epsilon Gaussian:

```text
X_t = sqrt(alpha_bar_t) * X + sqrt(1 - alpha_bar_t) * epsilon
loss = mean((epsilon_theta(X_t, t) - epsilon)^2)
```

- Cùng timestep cho cả clip.
- Noise độc lập giữa frame/pixel trong cấu hình primary.
- Cùng objective cho D2/DT.
- Không adversarial, identity, perceptual loss hoặc synthetic attack ở vòng đầu.

### 6.2. Recipe ban đầu

| Thành phần | Đề xuất |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Effective batch | 16 clip |
| Initial microbatch | 1-2 clip; chốt bằng probe |
| Accumulation | Đạt effective batch 16 |
| Gradient clipping | Norm 1 |
| EMA | 0,999 |
| Max epochs | 200 |
| Early stopping | Patience 20, sau ít nhất 20 epoch |
| LR schedule | Warmup 5 epoch, sau đó cosine |
| Numeric | BF16 nếu probe hỗ trợ; nếu không khóa FP32 trước launch |

Một epoch có số video-draw bằng số fitting video bonafide, sampling cân bằng
source. Cần định nghĩa chính xác partial batch/accumulation và optimizer-step
count trước execution; không dùng epoch label để che khác biệt budget.

Checkpoint/resume lưu model, EMA, optimizer, scheduler, RNG, sampling position
và hashes. Numerical failure/timeout không phải một completed model.

### 6.3. Augmentation

Proposal nhẹ và nhất quán cho toàn clip:

- Horizontal flip.
- Exposure khoảng +/-0,3 EV.
- Gamma 0,9-1,1.
- Channel gain 0,95-1,05.

Kiểm tra trực quan trên source trước khi seal. Không jitter màu độc lập từng
frame. Không đưa moiré/reflection/halftone vào bonafide training.

### 6.4. Checkpoint selection

Chọn EMA checkpoint có bonafide validation noise MSE thấp nhất:

- Equal-source mean.
- Window/timestep/noise validation cố định.
- Earliest checkpoint khi hòa.
- Không post-selection refit.

Denoising loss thấp không bảo đảm PAD tốt. Nếu chọn checkpoint bằng attacks
thì đó là study khác, không sửa claim strict one-class ngầm.

## 7. Scoring

### 7.1. Primary score

Không reverse diffusion 1.000 bước. Với mỗi clip:

1. Chọn ba mức alpha_bar gần 0,9; 0,5; 0,1 theo scheduler.
2. Mỗi mức dùng hai Gaussian draws được seed cố định.
3. Mỗi draw: thêm noise, một denoiser evaluation, đo epsilon MSE.
4. Trung bình qua draw/frame/channel/pixel để có e_j cho từng noise level.

Tổng cộng sáu denoiser evaluations mỗi clip, không phải latency đã đo.
Single-frame và multi-frame dùng cùng noise realization cho frame tương ứng.

Chuẩn hóa từng e_j bằng median/IQR từ bonafide source validation, equal-source
weighting, rồi lấy trung bình ba standardized scores.

- Score cao: bất thường hơn.
- Không tính lại normalization bằng target.
- IQR không hợp lệ phải báo lỗi hoặc blocked theo policy đã khóa, không fallback
  silent.
- Lưu residual maps và per-frame/per-level statistics để chẩn đoán.
- Chưa fusion thêm quantile, frequency hoặc region score ở primary study.

Pixel reconstruction MSE một bước và epsilon MSE liên hệ trực tiếp bằng hệ số
phụ thuộc timestep. Không coi chúng là hai nguồn bằng chứng độc lập.

### 7.2. Temporal diagnostics

Chạy DT trên:

- Clip đúng thứ tự.
- Clip xáo thứ tự, seed cố định.
- Một frame lặp thành clip.
- Bonafide clips dưới nuisance perturbations hợp lệ.

Các diagnostics không chọn primary recipe bằng target. Nhạy với shuffle không
tự chứng minh học được liveness vì shuffle cũng tạo chuyển động phi tự nhiên.
Residual temporal variation do injected noise không được diễn giải thành cue
spoof; luôn giữ noise generation được kiểm soát khi so sánh.

## 8. Baselines

| Baseline | Câu hỏi |
|---|---|
| D2 middle frame | Mốc single-image diffusion |
| D2 tổng hợp 16 frame | Lợi ích từ nhiều quan sát |
| DT clip | Giá trị bổ sung của temporal modeling |
| Denoising autoencoder nhỏ | Có cần diffusion nhiều mức noise không? |
| Frozen DINO + one-class score | So với representation baseline mạnh/rẻ |

Autoencoder baseline: cùng spatial backbone family, chỉ một noise level cố định
(proposal alpha_bar=0,5), dự đoán clean image, không variable timestep conditioning.
Train bonafide, cùng sampling/augmentation/selection discipline. Khóa topology,
loss và score trước launch. Đây là fixed-noise denoising baseline, không phải
đối chứng cho mọi dạng generative model.

DINO baselines:

- Frozen encoder với checkpoint/license/hash khai báo.
- Feature trung bình clip + shrinkage Gaussian fit trên bonafide fitting.
- Baseline temporal đơn giản: thêm thống kê chênh lệch feature frame liền kề.
- Cùng data splits và hai calibration regimes.

DINO dùng external pretraining nên không cùng điều kiện dữ liệu tuyệt đối.
Baseline này trả lời giá trị thực dụng, không cô lập riêng hiệu ứng pretraining.

Các kết quả supervised cũ chỉ là historical reference nếu chưa chạy matched
input/protocol. Không gọi đó là matched temporal comparison.

## 9. Thresholds và evaluation

### 9.1. Strict one-class

Đặt threshold từ bonafide source calibration, hướng đến source BPCER 5%;
sensitivity 1% và 10%. Proposal: chọn threshold đáp ứng bonafide acceptance của
source khó nhất.

Khóa quantile estimator, finite-sample policy và ties trước scoring. Báo số
video và tỷ lệ thực tế từng source. Không có đủ support thì báo thiếu bằng
chứng; không tuyên bố kiểm soát population error từ empirical quantile.

Bonafide threshold không chứng minh APCER an toàn.

### 9.2. Attack-aware source calibration

Với score và weights đã khóa:

- Primary attack FA <=5% trên từng source calibration.
- Sensitivity 1% và 10%.
- Accept khi S < threshold; ties reject.
- Dùng integer-count order-statistic threshold, bảo toàn source khó nhất.
- Không chọn score/model bằng calibration attacks.

Theo hướng score bất thường tăng dần, threshold an toàn chung không được vượt
threshold an toàn của bất kỳ source nào. Thiếu support/policy phải hiện rõ.

### 9.3. Metrics

Mỗi target/seed phải có:

- AUROC trên usable videos.
- Attack FA và bona-fide acceptance tại source-locked thresholds.
- APCER theo attack family khi đủ support và BPCER, kèm denominator.
- Technical-failure rate và failure reasons.
- Oracle BFA tại target FA 5% chỉ làm diagnostic.
- Latency/video, VRAM, CPU RAM và storage.

Giữ endpoint tham chiếu từ study trước: FA <=5% và BFA >=50%. Luôn báo integer
counts và uncertainty, không chỉ một cờ pass/fail.

Technical failures:

- Là nonaccept trong all-original operating metrics.
- Báo usable-only metrics riêng.
- Không tạo anomaly score giả để đưa vào AUROC.
- Không che việc FA thấp do pipeline từ chối phần lớn input.

So sánh cả common-usable subset lẫn all-original population để khác biệt
preprocessing availability không bị nhầm thành khả năng phân biệt của model.

### 9.4. Uncertainty

- 2.000 paired cluster-bootstrap repetitions.
- Subject khi biết; SiW theo video/content components với giới hạn subject
  independence được nêu rõ.
- Không bootstrap từng frame như các mẫu độc lập.
- Source thresholds được giữ cố định khi target bootstrap; CI là conditional,
  không bao gồm toàn bộ training/calibration uncertainty.
- Báo từng seed; không pool các seed thành test observations độc lập.

Target datasets đã được nghiên cứu trong repository nguồn. Kết quả mới là
developmental evidence, không gọi là untouched confirmation.

## 10. Generalization diagnostics

### 10.1. Nuisance sensitivity

Trên cùng source bonafide clip, đo score/BFA change dưới exposure, compression,
blur và crop offsets nhẹ có kiểm soát. Transform nhất quán theo thời gian.

Mức biến đổi cần pin trước scoring. Nếu biến đổi thu nhận hợp lệ gây tăng score
mạnh, đó là failure mode cần báo dù AUROC gộp tốt.

Có thể dùng source-held-out diagnostics để đánh giá domain sensitivity.
Nếu dùng chúng để chọn recipe, phải có inner selection protocol rõ và không
được gọi các observations đó là confirmation độc lập.

### 10.2. Duration

Giữ sampling 8 FPS:

- 1 giây: 8 frame.
- 2 giây: 16 frame.
- 3 giây: 24 frame.

Temporal convolution cho phép chiều dài input thay đổi, nhưng model train
2 giây được test 1/3 giây chỉ cung cấp length-transfer diagnostic. Không gọi
đó là matched optimal-duration study.

Primary 2 giây không thay đổi sau khi xem duration diagnostics. Báo eligibility
và paired common-video subset khi so duration.

### 10.3. Attack coverage

Báo theo attack family/domain. Unseen dataset không tự đồng nghĩa unseen attack
mechanism. Không hứa chống replay chất lượng cao chỉ vì model học temporal.

## 11. Các pha thực hiện

### A. Inventory và engineering

Chưa scientific training:

- Export source/split/support/eligibility counts.
- Pin private data bindings và policies.
- Test timestamp sampling, tracking, consistent augmentation và failure accounting.
- Test cache corruption, split isolation, deterministic scoring, checkpoint reload
  và resume state.
- Integrated probe decode -> crop -> forward -> backward -> scoring.
- Đo throughput, storage và memory trước bulk execution.

Điều kiện qua: đủ support, không known forbidden crossings, failure minh bạch,
tài nguyên trong budget. Không tự chuyển role hoặc bỏ failures để qua gate.

### B. Source-only pilot một outer context

Context held-out OULU, chỉ dùng CASIA/MSU/SiW source fitting/validation;
không mở OULU test.

Thử D2, DT, autoencoder để kiểm tra learning, numerical stability, overfit,
scoring repeatability, nuisance sensitivity và throughput.

Nếu thay recipe sau pilot, phải version lại và không đếm checkpoint cũ như
completed run của final recipe. Smoke/debug runs có inventory riêng.

### C. Study một seed

Seed đầu tiên đề xuất: 20260917.

| Model | Trainings trên 4 folds |
|---|---:|
| D2 | 4 |
| DT | 4 |
| Autoencoder | 4 |
| Tổng | 12 |

D2 single/multi-frame dùng chung checkpoint. DINO statistical fits và scoring
được ghi riêng, không gộp vào số neural trainings.

### D. Chốt số seed trước khi mở target

Nếu budget cho phép, dùng ba seeds: 20260917, 20260923, 20261001.
Giữ data split cố định qua seeds.

| Model | Trainings trên 4 folds x 3 seeds |
|---|---:|
| D2 | 12 |
| DT | 12 |
| Autoencoder | 12 |
| Tổng | 36 |

Nếu chỉ đủ một seed thì công bố exploratory study. Không xem target rồi chỉ
bổ sung seeds cho model đang thắng.

### E. Locked evaluation và publication

- Khóa code/config/checkpoint/scoring/threshold hashes.
- Score toàn bộ declared target test population một lần theo recipe.
- Separate-process replay từ saved checkpoints/scores để xác minh aggregate metrics.
- Public tất cả target/seed/regime, không chỉ best seed.
- Không tự fusion/retune hoặc đổi primary endpoint khi target thất bại.

## 12. Resource policy và deliverables

Sau integrated probe mới chốt GPU-hours, wall-clock, disk và memory caps.
Không suy ra runtime từ probe DINO lịch sử. Không launch full matrix khi GPU
availability hoặc source preprocessing cost chưa rõ.

Nếu budget không đủ, giảm scope trước launch và ghi amendment. Timeout,
partial fit hoặc thiếu target không được gọi completed study.

Deliverables dự kiến:

1. Versioned protocol/config và dependency lock.
2. Aggregate inventory/clip eligibility report.
3. Private clip manifests/cache, gắn provenance.
4. D2/DT/autoencoder implementation và focused tests.
5. Trainer/scorer có checkpoint resume, guards và explicit failures.
6. Baseline/calibration/evaluation runners dùng shared metric helpers khi phù hợp.
7. All-case tables, paired uncertainty, nuisance/duration diagnostics, resource report.
8. Reproducible checkpoint/scoring replay receipt.

Các mục trên chưa tồn tại trong repository này chỉ vì kế hoạch được commit.

## 13. Cách đọc kết quả và điều kiện kết luận

1. D2 multi-frame hơn D2 single-frame: bằng chứng lợi ích nhiều quan sát.
2. DT hơn D2 multi-frame: bằng chứng temporal modeling bổ sung giá trị.
3. Diffusion hơn fixed-noise autoencoder: bằng chứng cho multi-noise diffusion
   trong phạm vi đối chứng đã chạy, không phải mọi generative approach.
4. Cải thiện ở source-locked target operating points: liên quan khả năng sử dụng.

Oracle tốt hơn nhưng source-locked threshold vẫn thất bại chỉ cho thấy score
có headroom, chưa giải quyết threshold transfer.

Không kết luận thành công từ reconstruction đẹp, denoising loss thấp, temporal
shuffle sensitivity hoặc AUROC tăng đơn lẻ.

## 14. Các điểm cần phản biện trước implementation

1. 2 giây/16 frame/128x128 có giữ đủ cue, hay cần ưu tiên 256 ngay từ đầu?
2. Tracking/alignment có xóa chuyển động hữu ích hoặc tạo artifact theo dataset?
3. Bonafide support thực tế có đủ cho scratch diffusion và calibration không?
4. Strict one-class checkpoint criterion có phù hợp mục tiêu anomaly detection?
5. Ba noise levels và mean score có quá yếu, hay là baseline đủ minh bạch?
6. Fixed-noise autoencoder và DINO one-class có đủ mạnh để tránh kết luận dễ dãi?
7. Nên chạy 12 trainings exploratory hay chốt 36 trainings trước target access?
8. Với target đã được xem trong nghiên cứu cũ, bằng chứng nào còn thiếu để
   đưa ra claim generalization mạnh?

## 15. Related work và khác biệt

- [DiffusionAD](https://arxiv.org/abs/2303.08730): noise-to-normal reconstruction
  và segmentation cho anomaly detection; không chứng minh residual thuần
  generalize trong FAS.
- [AutoDDPM](https://arxiv.org/abs/2305.19643): khó khăn trong việc loại anomaly
  mà vẫn bảo toàn vùng bình thường.
- [DiffFAS, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07068.pdf):
  dùng diffusion để sinh attack phục vụ FAS, khác bonafide-only anomaly detection.

**Đề xuất cuối:** bắt đầu bằng inventory clip và integrated probe; sau đó so
D2/DT nhỏ với baselines, không bắt đầu bằng full video diffusion hoặc fusion
nhiều score chưa kiểm chứng.
