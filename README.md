# ASR_adapt

prepare dataset to be used for fineturning
ASR Fellowship Challenge Submission Report

Name: [Your Name]
Contact: [Your Email]

RESULTS:
- Base Model WER: [Calculate from base_transcriptions.txt]
- Fine-tuned Model WER: {results['best_wer']:.4f} ({results['best_wer']*100:.2f}%)
- WER Improvement: [Calculate difference]

MODEL INFORMATION:
- Base Model: badrex/w2v-bert-2.0-kinyarwanda-asr
- Total Parameters: 591,535,968
- Trainable Parameters: 11,010,048 (1.86%)
- Frozen Parameters: 580,525,920 (98.14%)

ADAPTER ARCHITECTURE:
- Method: LoRA (Low-Rank Adaptation)
- Target Modules: linear_q, linear_v, intermediate_dense, output_dense
- Rank (r): 16
- LoRA Alpha: 32
- Dropout: 0.1

TRAINING STRATEGY:
- Epochs: 3
- Batch Size: 16
- Learning Rate: 1e-3
- Optimizer: AdamW
- Gradient Clipping: 1.0

REPRODUCTION INSTRUCTIONS:
1. Install requirements: pip install transformers peft torch datasets evaluate
2. Load base model: badrex/w2v-bert-2.0-kinyarwanda-asr
3. Apply LoRA adapters with the provided configuration
4. Load adapter weights from 'best_adapter' directory
5. Run inference using the provided code

OBSERVATIONS:
[Add your observations about the training process and results]
"""
    
    with open("submission/report.pdf", "w") as f:
        f.write(report_content)
    
    print("✅ Submission package created in 'submission/' directory")
    print("📋 Please fill in the report template with your information")

# Create submission package
create_submission_package()