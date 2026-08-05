# Datasets

## customer_support_v3.parquet
Version: v1.4.2 (tracked by DVC)
Source: Internal CRM export + human annotations
Rows: ~48,000 training / 6,000 validation / 6,000 test
License: Internal proprietary
PII: Redacted (emails, phone numbers hashed)
Columns: ticket_id, text, category, priority, resolution, customer_tier, created_at

## product_docs_corpus
Source: Confluence + product manuals
Format: JSONL chunks with embeddings precomputed
Collection: product

## policy_kb
Source: Legal & compliance team
Format: Markdown files ingested into Chroma
Collection: policy
