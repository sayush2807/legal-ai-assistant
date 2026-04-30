# Legal AI Assistant: QLoRA Fine-Tuning + RAG over Legal Documents

Parameter-efficient fine-tuning of Mistral 7B and FAISS-based retrieval over Legal-BERT embeddings for legal holding prediction on LexGLUE CaseHOLD.

---

## Overview

Legal documents are long, domain-specific, and accuracy-critical — small misreadings of a holding can change the outcome of a case. This project explores two complementary approaches to legal reasoning: (1) parameter-efficient fine-tuning of Mistral 7B via QLoRA for generative holding prediction, and (2) grounding predictions in retrieved case law via FAISS over Legal-BERT embeddings, with three retrieval strategies compared on the same dataset.

Both pipelines are evaluated on LexGLUE CaseHOLD (45,000 train / 3,900 validation / 3,600 test), a benchmark of US federal court opinions where the task is to identify the correct holding from five candidates.

---

## Results

> **Important:** All inference runs below were interrupted before completion (Colab session limits). Metrics are from partial samples and should be read as preliminary, not full-dataset evaluations. Sample sizes are noted for each result.

### QLoRA Fine-Tuning — Mistral-7B-Instruct-v0.2

| Parameter | Value |
|-----------|-------|
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Quantization | 4-bit NF4, double quantization, bfloat16 compute |
| Memory footprint | ~3.5 GB (vs ~28 GB fp32 baseline) |
| LoRA rank / alpha / dropout | 64 / 16 / 0.03 |
| Trainable parameters | ~33.5M |
| Task type | CAUSAL_LM |
| Max sequence length | 384 tokens |
| Training examples | 400 (subset of 45,000-example train split) |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Batch size / gradient accumulation | 1 / 8 steps |
| Optimizer | paged_adamw_32bit |

**Training loss** (2 epochs, 100 steps, ~1 h 59 m on Colab T4):

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 1.6111 | 1.6241 |
| 2 | 1.4607 | 1.6379 |

**Inference evaluation** — *partial run: ~221–239 of 3,600 test examples*

| Metric | Embedding model | Value |
|--------|----------------|-------|
| Avg ROUGE-L | — | 0.2113 |
| Avg Cosine similarity | `all-mpnet-base-v2` | 0.5660 |
| Avg Cosine similarity | `nlpaueb/legal-bert-base-uncased` | 0.8962 |

---

### RAG Retrieval — Three Strategy Comparison

All three strategies use `nlpaueb/legal-bert-base-uncased` (768-dim) for retrieval embeddings, `faiss.IndexFlatL2` for the vector index, top-k = 3, and `mistralai/Mistral-7B-Instruct-v0.2` as the generation model. Evaluated on the validation split (3,900 examples); inference was interrupted in all three runs.

| Strategy | Index contents | Chunk size / overlap | FAISS vectors | Eval sample | ROUGE-L | Cosine (Legal-BERT) |
|----------|---------------|---------------------|---------------|-------------|---------|---------------------|
| **1 — Context only** (`02a`) | Context chunks from 1,000 train examples | 1,000 chars / 100 | 1,000 × 768 | ~21–29 of 3,900 | 0.1001 | 0.8648 |
| **2 — Context + holding** (`02b`) | Concatenated context + correct holding from all 45,000 train examples | 500 chars / 50 | (full train) | ~41–49 of 3,900 | 0.1560 | 0.8653 |
| **3 — Dual index** (`02c`) | Separate FAISS indexes for context chunks and holding chunks (1,000 train examples each) | 500 chars / 50 | 1,554 × 768 | ~41–49 of 3,900 | 0.0803 | 0.8491 |

**Note on cosine comparability:** Legal-BERT cosine scores (~0.85–0.87) reflect domain-tuned embedding similarity and are not comparable to the `all-MiniLM-L6-v2`-based cosine scores in `results/rag_eval_option3.csv` (~0.43), which used a different LLM (LLaMA 3 via Ollama) and a general-purpose embedding model.

---

## Stack

| Component | Library |
|-----------|---------|
| Fine-tuning | `transformers`, `peft`, `bitsandbytes`, `accelerate` |
| Dataset | `datasets` (HuggingFace) |
| Evaluation metrics | `evaluate`, `rouge-score`, `scikit-learn` |
| RAG retrieval | `faiss-cpu`, `langchain` (text splitter) |
| Retrieval embeddings | `sentence-transformers`, `transformers` |
| Data / utilities | `numpy`, `pandas` |

---

## Repo Structure

```
legal-ai-assistant/
├── notebooks/
│   ├── 01_qlora_finetune.ipynb              QLoRA fine-tuning (Mistral 7B, CaseHOLD)
│   ├── 02a_rag_context_only.ipynb           RAG strategy 1: context-only index
│   ├── 02b_rag_context_with_holdings.ipynb  RAG strategy 2: context + holding index
│   └── 02c_rag_dual_index.ipynb             RAG strategy 3: dual context + holdings indexes
├── src/
│   ├── config.py                            Embedding model and chunk settings
│   ├── data_loader.py                       Load .txt documents from a folder
│   ├── embed_store.py                       ChromaDB vector store utility (alternative implementation, not used in canonical notebooks 01/02a/02b/02c)
│   └── llm_pipeline.py                      RAG pipeline utility (alternative implementation, not used in canonical notebooks 01/02a/02b/02c)
├── data/
│   └── README.md                            Dataset download instructions
├── results/
│   └── rag_eval_option3.csv                 83-example eval (LLaMA 3 + MiniLM, preliminary)
├── requirements.txt
└── .gitignore
```

---

## Reproducing

1. **Download the dataset** — loaded automatically on first run:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("coastalcph/lex_glue", "case_hold")
   ```
   See `data/README.md` for split sizes and HuggingFace link.

2. **Set your HuggingFace token:**
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run notebooks in order** (GPU recommended for notebooks 01 and 02a–02c):
   - `notebooks/01_qlora_finetune.ipynb` — fine-tune Mistral 7B with QLoRA
   - `notebooks/02a_rag_context_only.ipynb` — build and evaluate RAG strategy 1
   - `notebooks/02b_rag_context_with_holdings.ipynb` — RAG strategy 2
   - `notebooks/02c_rag_dual_index.ipynb` — RAG strategy 3

   Notebooks were originally run on Google Colab with a T4 GPU. `bitsandbytes` 4-bit quantization requires CUDA; CPU-only runs will hit an error on the model load cell.

---

## Limitations and Future Work

- **Partial evaluation:** All inference runs were interrupted before completing the full validation/test sets. Reported metrics are from samples of 21–239 examples depending on the notebook. A full evaluation sweep is the most important next step.
- **Training scale:** QLoRA training used 400 of the 45,000 available training examples and 2 epochs; a full-data run with validation-based early stopping would substantially change the reported loss trajectory.
- **Retrieval query design:** All three RAG strategies queried the FAISS index with the fixed string `"What is the holding?"` rather than the actual case context. Embedding the test-case context as the query would be a meaningful improvement.

---

## Author

Ayush Srivastava — MS Data Science @ USF | Data Scientist Intern @ LexisNexis  
[linkedin.com/in/sayush2807](https://linkedin.com/in/sayush2807) · [sayush2807.github.io/portfolio](https://sayush2807.github.io/portfolio)
