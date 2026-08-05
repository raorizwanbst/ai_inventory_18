# Model Card: rag-classifier (LoRA fine-tune)

## Model Details
- Base model: meta-llama/Llama-3.1-8B-Instruct
- Fine-tune method: LoRA (r=16, alpha=32)
- Training run: lora-finetune-llama31-8b-v2
- MLflow run ID: (populated at train time)
- Registry: models:/rag-classifier/Production
- Artifact path: data/artifacts/lora-llama31-8b/final
- Framework: transformers + peft + torch

## Intended Use
Customer support response generation and ticket classification within Contoso RAG platform.

## Training Data
customer_support_v3.parquet (v1.4.2) — internal CRM tickets, PII redacted.

## Evaluation
- ROUGE-L: 0.41
- BERTScore F1: 0.78
- Exact match (category): 0.89

## Limitations
May hallucinate policy numbers if retrieval context is missing.
Not intended for medical, legal, or financial advice.
