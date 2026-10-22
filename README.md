# 📚 Research Intelligence RAG

A research-paper question answering system built with **Retrieval-Augmented Generation (RAG)**.

The system processes a research paper, creates page-aware text chunks, generates dense embeddings, stores them in a local Qdrant vector database, retrieves the most relevant passages for a question, and uses a lightweight instruction-tuned language model to generate an answer grounded in the retrieved context.

The project is being developed as a **learning-by-building AI/ML portfolio project**, with a focus on understanding each component of a modern RAG pipeline rather than relying on an end-to-end framework.

---

## ✨ Features

- 📄 PDF text extraction using **PyMuPDF**
- 🧩 Page-aware document chunking
- 🧠 Dense semantic embeddings using **BAAI/bge-base-en-v1.5**
- 🔎 Vector similarity search using **Qdrant**
- 🤖 Answer generation using **Qwen/Qwen2.5-0.5B-Instruct**
- 📌 Deterministic source references for retrieved chunks
- 📊 Retrieval and answer evaluation
- 🧪 Faithfulness and correctness evaluation using an LLM-based evaluator
- 🖥️ Interactive **Streamlit** interface
- 🧱 Modular project structure for future expansion

---

## 🏗️ Current Architecture

```text
Research Paper (PDF)
        │
        ▼
   PyMuPDF Loader
        │
        ▼
 Page-Aware Chunking
        │
        ▼
 BGE-base-en-v1.5
        │
        ▼
      Qdrant
        │
        ▼
   Top-K Retrieval
        │
        ▼
 Qwen2.5-0.5B-Instruct
        │
        ▼
 Answer + Sources
        │
        ▼
   Streamlit UI
```

The current implementation intentionally uses **one dense retrieval pipeline**.

It does not use BM25, reciprocal-rank fusion, query expansion, or a reranker in the main retrieval path. This keeps the system easier to understand, evaluate, and extend.

---

## 📂 Project Structure

```text
research-intelligence-rag/
│
├── data/
│   ├── raw/
│   │   └── <research-paper.pdf>
│   │
│   ├── processed/
│   │   └── evaluation.json
│   │
│   └── qdrant/
│       └── <local vector database>
│
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── retrieval/
│   │   ├── search.py
│   │   └── vector_store.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       └── faithfulness.py
│
├── tests/
│   └── test_faithfulness.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| PDF Processing | PyMuPDF |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Vector Database | Qdrant |
| Generation Model | Qwen/Qwen2.5-0.5B-Instruct |
| ML Framework | PyTorch |
| Transformers | Hugging Face Transformers |
| UI | Streamlit |
| Evaluation | Custom retrieval + LLM-based evaluation |

---

## 📄 Current Research Paper

The initial implementation was developed using a research paper on predicting drug response from single-cell expression data.

The source PDF is intentionally **not included in the repository**. The repository contains the code and evaluation resources required to reproduce and extend the system without redistributing the original paper.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vasisthhhhhh/research-intelligence-rag.git
cd research-intelligence-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Prepare the Research Paper

Place a research-paper PDF inside:

```text
data/raw/
```

The current ingestion pipeline expects a PDF to be available there.

The PDF is intentionally ignored by Git through `.gitignore`.

---

## ⚙️ Process the Paper

Run the ingestion pipeline:

```bash
python -m src.ingestion.pipeline
```

This extracts text from the PDF, creates page-aware chunks, and prepares the processed corpus.

---

## 🔎 Run Retrieval + Generation

The command-line interface can be used to ask questions about the indexed paper:

```bash
python -m src.retrieval.search
```

Enter a research question when prompted.

The system retrieves the most relevant chunks and passes them to the generation model.

---

## 🖥️ Run the Streamlit Application

Start the web interface:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

The interface provides:

1. A research question input
2. Retrieved context
3. A generated answer
4. Source pages and chunk identifiers
5. Expandable source evidence

---

## 📊 Evaluation

The project includes a small evaluation dataset containing questions, expected answers, and relevant chunk identifiers.

Run:

```bash
python -m src.evaluation.evaluate
```

The evaluation measures retrieval and generation performance using:

### Retrieval Metrics

- **Recall@5**
- **Precision@5**
- **Mean Reciprocal Rank (MRR)**

### Generation Metrics

- Semantic similarity
- Faithfulness
- Answer correctness

The faithfulness and correctness scores use a lightweight language-model-based evaluator and should therefore be treated as **heuristic evaluation signals rather than objective ground truth**.

---

## 🔬 Why RAG?

Large language models can generate fluent answers while lacking access to the specific information contained in a research paper.

RAG addresses this by separating the process into two stages:

```text
Retrieve relevant evidence
          ↓
Generate an answer from that evidence
```

This allows the model to answer questions using information retrieved directly from the source document instead of relying entirely on its pretrained knowledge.

The system also exposes the retrieved sources so that the user can inspect the evidence behind an answer.

---

## 🎯 Design Goals

The project focuses on understanding the underlying components of RAG rather than hiding them behind a framework.

The main goals are:

- Understand document ingestion
- Understand chunking strategies
- Understand dense embeddings
- Understand vector similarity search
- Understand retrieval quality
- Understand context construction
- Understand grounded generation
- Evaluate retrieval independently from generation
- Evaluate answer faithfulness
- Build a usable interface around the pipeline

---

## 🛣️ Roadmap

### Phase 1 — Single-Paper RAG ✅

- [x] PDF ingestion
- [x] Page-aware chunking
- [x] Dense embeddings
- [x] Qdrant vector storage
- [x] Top-K retrieval
- [x] LLM-based generation
- [x] Source display
- [x] Evaluation pipeline
- [x] Streamlit interface

### Phase 2 — Multi-Paper Research Assistant 🚧

- [ ] Support multiple research papers
- [ ] Add document-level metadata
- [ ] Identify the source paper for every retrieved chunk
- [ ] Improve document management
- [ ] Support adding new papers without rebuilding the entire project

### Phase 3 — Interactive Research Workspace

- [ ] Upload PDFs directly through Streamlit
- [ ] Automatically process uploaded papers
- [ ] Search across multiple papers
- [ ] Display document and page-level citations
- [ ] Improve answer formatting
- [ ] Add research-paper summaries

### Phase 4 — Advanced Research Intelligence

- [ ] Paper comparison
- [ ] Cross-paper question answering
- [ ] Citation graph exploration
- [ ] Research topic clustering
- [ ] Experimental result extraction
- [ ] Structured research insights

---

## 🧠 Key Learning Outcomes

This project is designed to demonstrate practical understanding of:

- Natural Language Processing
- Information Retrieval
- Semantic Search
- Vector Databases
- Embedding Models
- Large Language Models
- Retrieval-Augmented Generation
- LLM Evaluation
- Python ML Engineering
- Streamlit Application Development

---

## ⚠️ Current Limitations

The current implementation is intentionally small and serves as a baseline system.

Current limitations include:

- It is primarily designed around a single research paper.
- Retrieval quality is not yet optimized for all question types.
- The generation model is deliberately lightweight.
- LLM-based evaluation can be imperfect.
- The local Qdrant database is intended for development rather than production-scale deployment.
- The system does not yet provide a full multi-document research workspace.

These limitations are part of the planned development roadmap rather than hidden implementation details.

---

## 📌 Project Status

**Status:** 🚧 Active Development

**Current milestone:** Single-paper RAG system with dense retrieval, grounded generation, evaluation, and Streamlit interface.

The next major milestone is transforming the current single-paper implementation into a **multi-paper research intelligence system**.

---

## 👨‍💻 Author

**Nitin Pandey**

BCA — Artificial Intelligence & Machine Learning

Built as a practical exploration of Retrieval-Augmented Generation, information retrieval, vector databases, and research-focused AI systems.