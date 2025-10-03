# CV8501 Assignment 2: Skin Lesion Classification with CNNs and VLMs (LLaVA-Med)

This repository contains code and experiments for **skin lesion classification** on the **HAM10000** dataset.  
We compare:

1. **CNN baseline** (ResNet18 trained from scratch)  
2. **Pretrained LLaVA-Med** (zero-shot VLM)  
3. **Fine-tuned LLaVA-Med** (task-specific adaptation)

---

## 📦 Setup

### 1. Clone this repository
```bash
git clone https://github.com/Mashrafi27/CV8501_A2.git
cd CV8501_A2
````

### 2. Environment

We recommend using conda:

```bash
conda create -n llava python=3.10
conda activate llava
pip install -r requirements.txt
```

Additional dependencies:

* [PyTorch](https://pytorch.org/get-started/locally/) with CUDA
* Hugging Face `transformers`, `datasets`
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)

### 3. Clone **LLaVA-Med**

Evaluation and fine-tuning require the **LLaVA-Med** repo:

```bash
git clone https://github.com/microsoft/llava-med.git
cd llava-med
pip install -e .
```

⚠️ Important: If you also cloned the original [LLaVA repo](https://github.com/haotian-liu/LLaVA), ensure **LLaVA-Med takes precedence** on your `PYTHONPATH`. Both repos define `llava/*`.

### 4. Pretrained Checkpoint

We use the Hugging Face model:
`liuhaotian/llava-med-v1.5-mistral-7b`

Download automatically:

```bash
huggingface-cli download liuhaotian/llava-med-v1.5-mistral-7b
```

Or directly reference in scripts:

```bash
--model-path liuhaotian/llava-med-v1.5-mistral-7b
```

---

## 📊 Dataset

We use the **HAM10000** dermatoscopic dataset:

* ~10,000 images across **7 classes**:
  `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`

* Splits (following assignment protocol):

  * **Train**: ~7,000 images (70%)
  * **Validation**: 1,503 images (15%)
  * **Test**: 1,503 images (15%)

### VQA Formulation

For LLaVA-Med, each image is framed as a **closed-ended VQA**:

```
What is the lesion type? Choose one of: akiec, bcc, bkl, df, nv, mel, vasc.
```

The output is normalized into one of the 7 labels with a regex-based parser.
The CNN baseline uses standard multiclass classification.

---

## 🚀 Training & Evaluation

### 1. CNN Baseline

Train ResNet18 from scratch with cross-entropy:

```bash
python src/train_cnn.py --data /path/to/HAM10000 --epochs 50 --batch-size 32
```

Validation metrics (peak):

* **Accuracy** ≈ 0.88
* **Macro-F1** ≈ 0.79
* **Macro-AUC** ≈ 0.98

Final test metrics are reported in the paper.

---

### 2. LLaVA-Med Pretrained (zero-shot)

Run inference on test split:

```bash
python llava/eval/model_vqa.py \
  --model-path liuhaotian/llava-med-v1.5-mistral-7b \
  --question-file /path/to/test_vqa.jsonl \
  --image-folder /path/to/images \
  --answers-file runs/test_predictions_pretrained.jsonl \
  --conv-mode mistral_instruct \
  --temperature 0.0
```

---

### 3. LLaVA-Med Fine-tuning

Fine-tune with LoRA + DeepSpeed:

```bash
deepspeed llava/train/train.py \
  --model_name_or_path liuhaotian/llava-med-v1.5-mistral-7b \
  --data_path /path/to/train_instruct.jsonl \
  --image_folder /path/to/images \
  --output_dir checkpoints/llava-med-finetuned \
  --lora_enable True --lora_r 128 --lora_alpha 256 \
  --num_train_epochs 1 --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 \
  --deepspeed ./scripts/zero3.json \
  --learning_rate 2e-4 --mm_projector_lr 2e-5 \
  --weight_decay 0.0 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --save_strategy steps \
  --save_steps 50000 --save_total_limit 1 \
  --logging_steps 1 --bf16 True --tf32 True
```

---

### 4. Scoring

Evaluate predictions against ground truth:

```bash
python src/evaluation/score_closed_vqa_full.py \
  --questions data/test_vqa.jsonl \
  --answers runs/test_predictions.jsonl \
  --out-csv results/per_sample.csv \
  --out-confusion results/confusion.csv
```

Metrics reported: **Accuracy**, **macro-F1**, **macro-AUC (OvR)**.
Confusion matrices are also generated from the CSV.

---

## 📈 Results (Summary)

| Model                  | Accuracy | Macro-F1 | Macro-AUC |
| ---------------------- | -------- | -------- | --------- |
| CNN (baseline)         | **0.88** | **0.79** | **0.98**  |
| LLaVA-Med (pretrained) | 0.12     | 0.06     | 0.52      |
| LLaVA-Med (fine-tuned) | 0.72     | 0.31     | 0.59      |

---

## ⚠️ Notes & Issues

* **Repo conflicts:** Both LLaVA and LLaVA-Med define `llava/model/builder.py`.
  Ensure **LLaVA-Med** is installed in editable mode (`pip install -e .`) and comes first in `PYTHONPATH`.

* **GPU memory:** LLaVA-Med nearly saturates 40GB A100 GPUs.
  For smaller GPUs, reduce batch size or use gradient checkpointing.
  Evaluation on CPU is possible but extremely slow.

* **Answer normalization:** Free-text outputs (e.g. “melanocytic nevus”) are mapped into class tokens using regex rules, with negation handling.

---

## 📜 Reproducibility & Code Availability

* All scripts for training, evaluation, and scoring are in this repository.
* Full experiment logs (loss curves, validation metrics, confusion matrices) are included in the report.
* For complete instructions, see the [README](https://github.com/Mashrafi27/CV8501_A2.git).

---

## 📚 References

* Tschandl, P., et al. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*.
* Liu, H., et al. (2023). *LLaVA: Large Language and Vision Assistant*.
* Microsoft Research. *LLaVA-Med: Towards Large Multimodal Models for Medicine*.

