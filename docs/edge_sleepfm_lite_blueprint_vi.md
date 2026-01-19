# Blueprint triển khai Edge SleepFM-Lite (độ chính xác tối đa + mobile tối đa)

Tài liệu này tổng hợp blueprint triển khai **Edge SleepFM-Lite** theo định hướng:

* **Accuracy tối đa**: teacher mạnh, distillation đúng cách.
* **Mobile tối đa**: model nhỏ, INT8/QAT, streaming.
* **Dùng Spark đúng chỗ**: ETL/windowing/feature-store, không lạm dụng Spark cho training DL.

Nội dung được viết để **triển khai trên Google Colab** và bám theo pipeline có sẵn trong **sleepfm-clinical** (preprocessing EDF→HDF5, pretrain, generate embeddings, finetune staging, checkpoint). Tài liệu này **không sao chép mã nguồn** của repo gốc mà cung cấp kế hoạch/khung triển khai có thể bắt tay code ngay.

---

## 0) Tổng quan pipeline (Control Plane / Data Plane)

**Data Plane (Spark)**

* Dọn dữ liệu, chuẩn hóa, cắt epoch 30s, resample, tạo “mobile-view”, lưu thành Parquet/Delta.
* Chạy các job: **Index & Split**, **Epoch Store Builder**, **Teacher-Target Materialization**.

**Teacher Plane (PyTorch)**

* Dùng pretrained base model làm điểm xuất phát.
* DAP (Domain-Adaptive Pretraining) trên dữ liệu mục tiêu.
* Multi-task finetune (sleep staging + OSA).

**Student Plane (PyTorch)**

* SleepFM-Lite student (DS-CNN/TCN hoặc tiny transformer) + distillation.
* QAT/INT8 và (tùy chọn) pruning.

**Mobile Plane (TFLite + NNAPI)**

* Xuất TFLite INT8, streaming inference, đo latency/RAM/size.

---

## 1) Chuẩn dữ liệu & nhãn

### 1.1 Nhãn đầu ra

* **Sleep staging**: 5 lớp (W, N1, N2, N3, REM), epoch 30s.
* **OSA screening**:
  * Binary: AHI ≥ 5 (hoặc ≥ 15) vs < ngưỡng.
  * Hoặc 3 lớp: <5 / 5–15 / ≥15.

### 1.2 “Mobile-view modalities”

* **EEG (1–2 kênh)**: mô phỏng headband/wearable.
* **PPG**: nếu PSG có pleth/SpO2 waveform, dùng làm proxy PPG.
* **ACC**:
  1. Bỏ ACC (EEG+PPG) — sạch, thực tế.
  2. Tạo ACC-proxy từ tín hiệu chuyển động/artefact (EMG/belt/position) — ghi rõ là proxy.

---

## 2) Spark dùng đúng chỗ (ETL/windowing/feature-store)

### 2.1 Lưu trữ 3 tầng

* **Raw**: EDF + labels.
* **Canonical**: HDF5 theo chuẩn pipeline (để tương thích sleepfm-clinical).
* **Training Store**: Parquet/Delta (đã epoching + mobile-view) để training nhanh.

### 2.2 Spark Job A — Index & Split (tránh leakage)

**Mục tiêu**: tạo bảng metadata và split theo subject.

* Input: danh sách EDF/label.
* Output: bảng metadata `subject_id, night_id, path_edf, path_labels, sampling_rates, modalities_available`.
* Split **theo subject** (train/val/test), lưu JSON/Parquet.

**Gợi ý logic split (Spark SQL)**

* Tính `subject_hash` → chia theo phần trăm.
* Bảo đảm các night của cùng subject rơi vào cùng split.

### 2.3 Spark Job B — Epoch Store Builder (nơi Spark “ăn tiền”)

**Mục tiêu**: đọc HDF5, cắt epoch 30s, chuẩn hóa, lọc, và xuất Parquet.

* Lọc EEG: 0.5–40Hz + notch 50Hz.
* Resample về sampling rate chuẩn.
* Epoching 30s.

**Schema gợi ý (Parquet)**

* keys: `subject_id, night_id, epoch_idx`
* inputs:
  * `eeg`: `array<float>` shape `[C_eeg, T_eeg]`.
  * `ppg`: `array<float>` shape `[1, T_ppg]`.
  * `acc`: `array<float>` shape `[3, T_acc]` hoặc null.
  * `mask_modality`: bitmask missing/dropout.
* labels:
  * `y_stage`: int (0..4)
  * `y_osa`: int (0/1 hoặc 0/1/2)

### 2.4 Spark Job C — Teacher-Target Materialization

**Mục tiêu**: chạy teacher inference offline để sinh target distillation và join vào Parquet.

* Teacher outputs: `t_embed`, `t_logits_stage`, `t_logits_osa`.
* Join theo keys `subject_id, night_id, epoch_idx`.
* Student training đọc trực tiếp Parquet → data-loader nhẹ.

---

## 3) Teacher: tối đa hóa chất lượng

### 3.1 Start point

* Dùng pretrained base model làm điểm xuất phát.
* Không pretrain-from-scratch trừ khi có dataset rất lớn.

### 3.2 DAP (Domain-Adaptive Pretraining)

* Input: PSG thật của bạn (càng đa modality càng tốt).
* Objective: giữ kiểu contrastive/self-supervised theo SleepFM.
* Khuyến nghị:
  * LR 1e-5 ~ 5e-5, warmup + cosine decay.
  * Modality/channel dropout để tăng robustness.

### 3.3 Multi-task Fine-tune Teacher

* Head 1: sleep staging (CE, class-balanced).
* Head 2: OSA severity (weighted CE hoặc focal).
* Optional: head artifact/noise.
* Dùng context 2–5 phút (sequence of epochs) để nắm transition.

---

## 4) Student: SleepFM-Lite (mobile-first)

### 4.1 Kiến trúc khuyến nghị

**Student-A (khuyến nghị sản phẩm)**

* EEG branch: Depthwise-Separable Conv1D + TCN (dilated).
* PPG branch: tương tự nhưng nhỏ hơn.
* ACC branch: optional.
* Fusion: concat → 1–2 FC.
* Mục tiêu 0.5–2M params.

**Student-B (upper-bound)**

* CNN stem → 2–4 Transformer-lite/Conformer blocks.
* Dùng khi cần thêm accuracy, nhưng QAT gần như bắt buộc.

### 4.2 Distillation “đúng cách”

Loss tổng:

```
L = λ_sup L_CE + λ_KL L_KL + λ_emb L_emb + λ_feat L_feat
```

* **Logit distillation** (KL + temperature 2–4) cho staging & OSA.
* **Embedding alignment** (cosine/MSE) giữa student/teacher.
* **Intermediate feature matching** (nếu có).

**Trick để vừa mạnh vừa bền**

* Modality dropout khi train student.
* Fake-quant noise ngay trước QAT để chịu INT8.

### 4.3 Lịch training gợi ý

* S1: supervised 5–10 epochs.
* S2: bật distillation 20–50 epochs.
* S3: fine-tune với augmentation mạnh + calibration threshold.

---

## 5) Mobile tối ưu: INT8 + QAT + pruning

### 5.1 QAT

* Bước 1: student FP32 tốt nhất.
* Bước 2: QAT 5–20 epochs, LR nhỏ (1e-5~1e-4).
* Bước 3: export TFLite INT8.

### 5.2 Structured pruning

* Channel pruning 10–30%.
* Fine-tune lại kèm distillation.

### 5.3 TFLite + NNAPI

* Export TFLite INT8 (static shapes).
* Android: bật NNAPI delegate nếu có.

### 5.4 Streaming inference

* Input 30s/epoch.
* Nếu dùng context: cache feature hoặc dùng state (TCN/GRU-lite).

---

## 6) Benchmark & tiêu chí “tốt nhất”

### 6.1 Offline metrics

* Staging: Macro-F1, Cohen’s Kappa.
* OSA: AUROC, AUPRC, Sensitivity @ Specificity=0.90.
* Ablations:
  * EEG-only vs EEG+PPG vs EEG+PPG+ACC.
  * Missing modality stress test.
  * Channel drop stress test.

### 6.2 On-device metrics (edge bắt buộc)

* Model size (MB, INT8).
* RAM peak.
* Latency/epoch (ms) trên 2 nhóm thiết bị.
* Energy proxy (chạy liên tục 30 phút, xem thermal throttling).

---

## 7) Deliverables

1. Spark pipelines: epoch store + teacher target materialization.
2. Teacher: pretrained + DAP + multi-task finetune.
3. Student-Lite: distill + QAT INT8 + (optional) pruning.
4. Mobile demo: TFLite streaming inference + hiển thị staging + OSA risk.
5. Báo cáo: accuracy + benchmark + ablation.

---

## 8) Colab runbook (khung triển khai chi tiết)

> Lưu ý: Đây là **khung triển khai** (template) để bạn bắt tay code ngay. Bạn cần thay các đường dẫn dữ liệu và cấu hình model cho phù hợp.

### 8.1 Setup môi trường

```bash
# Colab cell
!pip -q install torch torchvision torchaudio
!pip -q install pyspark==3.5.1
!pip -q install h5py numpy pandas pyarrow delta-spark
```

### 8.2 Mount dữ liệu & phân tầng

```bash
# Colab cell
from google.colab import drive

drive.mount('/content/drive')
RAW_DIR = "/content/drive/MyDrive/psg/raw"
HDF5_DIR = "/content/drive/MyDrive/psg/hdf5"
PARQUET_DIR = "/content/drive/MyDrive/psg/parquet"
SPLIT_PATH = "/content/drive/MyDrive/psg/splits"
```

### 8.3 Job A — Index & Split (Spark)

```python
# Colab cell
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("sleepfm-index-split") \
    .getOrCreate()

# Ví dụ: load metadata CSV do bạn tự tạo (path_edf, path_labels, subject_id, night_id, ...)
meta_df = spark.read.csv(f"{RAW_DIR}/metadata.csv", header=True)

# Hash subject_id để split theo subject
from pyspark.sql.functions import sha2, col

meta_df = meta_df.withColumn("subject_hash", sha2(col("subject_id"), 256))

# Ví dụ: split theo prefix hash (đơn giản, bạn có thể thay logic)
train_df = meta_df.filter(col("subject_hash").substr(1, 2) <= "aa")
val_df = meta_df.filter((col("subject_hash").substr(1, 2) > "aa") & (col("subject_hash").substr(1, 2) <= "cc"))
test_df = meta_df.filter(col("subject_hash").substr(1, 2) > "cc")

train_df.write.mode("overwrite").parquet(f"{SPLIT_PATH}/train")
val_df.write.mode("overwrite").parquet(f"{SPLIT_PATH}/val")
test_df.write.mode("overwrite").parquet(f"{SPLIT_PATH}/test")
```

### 8.4 HDF5 canonical (preprocessing EDF→HDF5)

```bash
# Colab cell (clone repo gốc và chạy preprocessing)
!git clone https://github.com/zou-group/sleepfm-clinical.git
%cd sleepfm-clinical

# Chạy preprocessing, trỏ input/output đến RAW_DIR/HDF5_DIR đã mount
!python preprocessing/preprocessing.py --input_dir "${RAW_DIR}" --output_dir "${HDF5_DIR}"
```

Kiểm tra keys HDF5 thực tế và đối chiếu với các key mong đợi:

```python
import h5py
from pathlib import Path

expected_keys = {
    "signals/eeg",
    "signals/ppg",
    "signals/acc",
    "labels/stage",
    "labels/osa",
}

hdf5_path = Path(HDF5_DIR) / "your_file.h5"  # TODO: đổi đúng tên file
with h5py.File(hdf5_path, "r") as f:
    actual_keys = set()

    def collect_keys(name, obj):
        if isinstance(obj, h5py.Dataset):
            actual_keys.add(name)

    f.visititems(collect_keys)

missing = expected_keys - actual_keys
extra = actual_keys - expected_keys
print("Missing keys:", missing)
print("Extra keys:", extra)
```

### 8.5 Job B — Epoch Store Builder (Spark)

```python
# Colab cell (pseudo-code)
# Bạn cần tự implement reading HDF5 -> epoching -> write Parquet.
# Spark có thể gọi UDF để xử lý từng file HDF5.
```

### 8.6 Teacher DAP + Multi-task Finetune (PyTorch)

```bash
# Colab cell (pseudo-code)
# python sleepfm/pipeline/pretrain.py --config configs/dap.yaml
# python sleepfm/pipeline/finetune_sleep_staging.py --config configs/finetune_multitask.yaml
```

### 8.7 Job C — Teacher-Target Materialization

```python
# Colab cell (pseudo-code)
# 1) Dataloader đọc Parquet -> PyTorch teacher inference
# 2) Lưu output t_embed/t_logits_* ra Parquet
# 3) Spark join vào training store
```

### 8.8 Student distillation + QAT

```bash
# Colab cell (pseudo-code)
# python student/train_distill.py --config configs/student_distill.yaml
# python student/train_qat.py --config configs/student_qat.yaml
```

### 8.9 Export TFLite INT8

```python
# Colab cell (pseudo-code)
# Convert PyTorch -> ONNX -> TFLite (hoặc TorchScript -> TFLite via converter)
```

---

## 9) Checklist để đạt “best” (accuracy + mobile)

* Teacher mạnh: DAP + multi-task + context.
* Student: distillation (logit + emb + feat) + modality dropout.
* QAT INT8 + (optional) pruning.
* Spark chỉ dùng cho ETL/windowing/joins.
* Benchmark thiết bị thật (latency/RAM/size).

---

## 10) Gợi ý cấu trúc repo nội bộ

```
edge-sleepfm-lite/
  data/
  etl/
    spark_jobs/
  teacher/
    configs/
    train/
  student/
    configs/
    train/
  mobile/
    tflite/
    android_demo/
  docs/
    edge_sleepfm_lite_blueprint_vi.md
```

---

## 11) Ghi chú đạo đức & báo cáo

* Mô tả rõ **proxy modalities** (nếu dùng).
* Báo cáo leakage control (split theo subject).
* Báo cáo on-device metrics khi nói “edge”.
