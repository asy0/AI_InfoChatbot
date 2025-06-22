# AI InfoBot – Semantic Search Update Protocol
#### by Niklas Brandtner
## Purpose

This document outlines recent updates made to the AI InfoBot prototype.
The goal of the system is to semantically search university-related documents (e.g., *Studienordnung*, *Datenschutzrichtlinien*) and return exact matching sentences or paragraphs with source citation, **without generating new content**.

---

## Summary of Updates

### 1. Switched to Paragraph-Based Embeddings
- Added `extract_paragraphs(text)` to `nlp_processing.py`
- Splits raw PDF text into **paragraphs** instead of isolated sentences
- Provides better **semantic context** for embedding and search
- Improves accuracy for multi-line concepts such as regulations, explanations, or legal phrasing

### 2. Preserved Sentence Splitting for Testing
- Existing sentence extraction using `spaCy` kept under `extract_sentences(text)`
- Allows switching between **sentence-level and paragraph-level** retrieval by commenting/uncommenting in `main.py`

### 3. Improved Semantic Search Logic
- Uses pre-trained multilingual Sentence-BERT model:
  `distiluse-base-multilingual-cased-v2`
- All document chunks (sentences or paragraphs) are **embedded once and cached** using `pickle`
- User queries are expanded using a **customizable synonym list** in `query_expansion.json` before encoding

### 4. Added actual Chat Feature
- Uses input() function to let user enter question, then uses that question to search embedded sentences/paragraphs

---

## How It Works

### Step 1: Document Processing
- PDFs are read using **PyMuPDF**
- Text is cleaned using `clean_text()`
- Text is split into either:
  - Sentences (via `spaCy`), or
  - Paragraphs (via custom rule-based splitter)
- Each chunk is encoded into a **semantic vector** using `sentence-transformers`

### Step 2: Query Handling
- User enters a **natural-language question**
- Query is expanded using terms from `query_expansion.json`
- Expanded query is embedded into a vector

### Step 3: Retrieval
- Query vector is compared against all document vectors using **cosine similarity**
- System returns the **top-k most relevant chunks** with:
  - Source filename
  - Similarity score

---

## Files Involved

| File                 | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `main.py`            | Controls processing flow, loads cache, handles user input and results       |
| `pdf_reader.py`      | Extracts and cleans text from PDF files                                     |
| `nlp_processing.py`  | Contains logic for sentence/paragraph extraction, embedding, and search     |
| `query_expansion.json` | Stores keyword–synonym mappings to improve semantic matching               |

---

