# 🧾 Legal Document Q\&A Assistant

A lightweight Streamlit application designed for legal professionals, researchers, and admin staff to extract and query scanned or digital legal documents (PDFs/images) using OCR, embeddings, and Claude 3.

---

## 🚀 Features

* 🔍 **OCR-Powered Text Extraction**

  * Extracts text from scanned PDFs or images using PyMuPDF + Tesseract

* 🤖 **LLM-Powered Q\&A System**

  * Uses Claude 3 Sonnet via LangChain to answer queries contextually

* 🧠 **Local Semantic Search**

  * Vectorizes document chunks using HuggingFace embeddings and FAISS

* 💡 **Simple & Intuitive UI**

  * Drag-and-drop Streamlit interface for easy file uploads and question answering

---

## ⚙️ Tech Stack

| Layer       | Tech Stack                            |
| ----------- | ------------------------------------- |
| OCR         | `PyMuPDF`, `pytesseract`, `pdf2image` |
| Embeddings  | `HuggingFace Transformers` (MPNet)    |
| Vector DB   | `FAISS` (local, in-memory)            |
| LLM Backend | `Claude 3 Sonnet` via `LangChain`     |
| Interface   | `Streamlit`                           |
| Language    | Python 3.10+                          |

---

## 📥 Installation

```bash
# Clone the repository
https://github.com/your-username/legal-doc-qa.git
cd legal-doc-qa

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Set your Anthropic API key as an environment variable or within Streamlit secrets.

```bash
export ANTHROPIC_API_KEY="your_claude_api_key_here"
```

Or inside `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your_claude_api_key_here"
```

---

## 🖼️ Usage

```bash
streamlit run app.py
```

1. Upload a scanned PDF or image file (jpg/png).
2. The system extracts the text and embeds it using FAISS.
3. Ask your legal question.
4. Claude 3 will respond using only the document’s content.

---

## 📌 Sample Inputs

* Indian court judgments
* Scanned FIR documents
* Legal case bundles
* Contractual agreements (PDF/image)

---

## 📦 Deployment Options

* ✅ **Local VM:** Suitable for law firms with privacy constraints.
* ☁️ **AWS EC2:** Scalable and secure cloud deployment.

---

## 🙌 Contributions

PRs and feedback welcome!

---

## 📄 License

MIT License
