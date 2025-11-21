# Adapter-Based Fine-Tuning for Low-Resource ASR:
### Domain Adaptation on Kinyarwanda Health Dataset 

**Author**: Elvis Tata Tanghang  
**Contact**: getson2215@gmail.com  
**Date**: November 2025

## Abstract

This project implements an adapter-based fine-tuning strategy applied to the pretrained `badrex/w2v-bert-2.0-kinyarwanda-asr` model (originally fine-tuned on 1000 hours of general domain data). We adapt this model to the **health domain** using **Low-Rank Adaptation (LoRA)**, a parameter-efficient fine-tuning (PEFT) technique. By training only **1.86%** of the model parameters (approx. 11M), we successfully aligned the model to the health domain while avoiding catastrophic forgetting.

## Demo 
https://huggingface.co/spaces/ElvisTata2024/Kinyarwanda-Health-ASR

**Key Result**: We improved the Word Error Rate (WER) from **6.51%** (base) to **6.45%** (adapted), demonstrating a **0.06%** improvement while maintaining the model's general robustness.

## 1. Methodology

### 1.1 Adapter Architecture (Wav2Vec2-BERT + LoRA)

We froze the massive 580M parameter encoder of the base model to preserve its feature extraction capabilities and injected LoRA matrices into the Transformer layers.

- **Target Modules**: All linear layers in the Transformer block:
  - Attention: `linear_q`, `linear_k`, `linear_v`, `linear_out`
  - Feed-Forward: `intermediate_dense`, `output_dense`
- **Configuration**: Rank ($r$) = 16, Alpha ($\alpha$) = 32, Dropout = 0.1

| Component | Parameters (Millions) | Trainable % |
|-----------|-----------------------|-------------|
| Base Model (Frozen) | 580 M | 0.00% |
| Adapter Module (Trainable) | 11 M | 100% |
| **Total** | **591 M** | **1.86%** | 

### 1.2 Training Strategy

- **Dataset**: Kinyarwanda Health dataset (`afrivoice-kinyarwanda-health`)
- **Data Loading**: Custom `IterableDataset` for efficient streaming of compressed `.tar.xz` archives.
- **Preprocessing**: 
  - Audio resampled to 16kHz.
  - Text normalized to lowercase `a-z`, space, and apostrophe (`'`).
  - Silent/corrupt audio (< 1e-5 amplitude) filtered out.
- **Hyperparameters**:
  - Learning Rate: 5e-5
  - Batch Size: 1 (Effective batch size via accumulation)
  - Gradient Accumulation: 4
  - Max Steps: 4000
  - Precision: FP16 (Mixed Precision)

## 2. Experiments & Results

We evaluated the models on the validation split using Word Error Rate (WER) as the primary metric. We also utilized a 2-gram KenLM for decoding.

### 2.2 Results Table

| Setup | Greedy WER ⬇ | WER + KenLM (2-gram) ⬇ | KenLM Improvement ⬆ |
|-------|--------------|------------------------|---------------------|
| **Base Model** | 6.51% | 6.03% | 0.48% |
| **Adapted Model** | **6.45%** | **5.98%** | 0.47% |
| **Improvement** | **0.06%** | **0.04%** | 0.02% |

The reduction in WER validates that the adapters successfully learned domain-specific phonetic patterns of the health dataset without altering the generalized acoustic features of the base model.

## 3. Reproducibility Instructions

### Environment Setup
```bash
pip install transformers datasets torch librosa evaluate jiwer peft bitsandbytes pyctcdecode
# Install KenLM
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Data Preparation
The `Final_ASR_Adapt.ipynb` notebook handles:
1. Downloading the `afrivoice-kinyarwanda-health` dataset.
2. Preprocessing audio to 16kHz.
3. Cleaning text labels.

### Training
Run the training script/notebook cell. It initializes the `Wav2Vec2BertForCTC` model, attaches the `LoraConfig`, and executes the Trainer.

### Evaluation
Rune the evaluation cell on the notebook to calcuare the WER and CER of the base model and finetuned model


## 4. References

[1] Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv preprint arXiv:2106.09685.


[2] B. Thomas, S. Kessler, and S. Karout, “Efficient Adapter Transfer of
Self-Supervised Speech Models for Automatic Speech Recognition,”
arXiv:2202.03218 (2022).

[3] W. Hou et al., “Exploiting Adapters for Cross-lingual Low-resource Speech
Recognition,” arXiv:2105.11905 (2021).


[4] N. Houlsby et al., “Parameter-Efficient Transfer Learning for NLP,”
arXiv:1902.00751 (2019).
