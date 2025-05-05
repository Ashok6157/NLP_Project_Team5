# Suicide Detection and Mental Health Q&A using NLP and LLMs

This project implements a two-stage Retrieval-Augmented Generation (RAG) system for mental health support. It detects suicidal ideation in social media text and provides empathetic, context-aware responses using open-source large language models (LLMs).

---

## Project Objectives

- Detect suicidal intent from user-generated text (Reddit-style posts).
- Retrieve similar real-world posts using sentence embeddings.
- Generate emotionally intelligent, factual responses using LLMs.
- Evaluate LLMs on factuality, relevance, and empathy.

---

## System Architecture

The pipeline consists of 5 core modules:

1. **Text Loading** – Import and clean Reddit-style user post data.  
2. **Embedding** – Generate semantic vectors using `all-MiniLM-L6-v2`.  
3. **Vector Store** – Store embeddings in `ChromaDB` for similarity search.  
4. **Retriever** – Retrieve top-k similar posts using FAISS-based search.  
5. **LLM Generation** – Use LLMs to generate empathetic answers to predefined mental health questions.

---

## LLMs Used

| Model           | Description                                  |
|------------------|----------------------------------------------|
| `FLAN-T5-Base`    | Lightweight encoder-decoder, fast but shallow. |
| `LaMini-Flan-T5`  | Compact, instruction-tuned model with low latency. |
| `Alpaca-7B`       | Instruction-tuned LLaMA model, strong in reasoning and empathy. |

All models were deployed via HuggingFace pipelines and evaluated within LangChain's `RetrievalQA` framework.

---

## Evaluation Metrics

Models were assessed on:
- **Factual Accuracy** – Is the answer grounded in retrieved evidence?
- **Relevance** – Does the answer directly respond to the user query?
- **Empathy** – Is the tone appropriate for mental health scenarios?

---

## Average Evaluation Scores

| Model        | Factual Accuracy | Relevance | Empathy |
|--------------|------------------|-----------|---------|
| Flan-Alpaca  | 1.87             | 1.87      | 2.00    |
| Flan-T5      | 1.40             | 1.53      | 1.20    |
| LaMini       | 1.53             | 1.60      | 1.67    |

---

## Visualization

![image](https://github.com/user-attachments/assets/1c89f2c7-aa18-482e-8e6f-534ab78715ca)


---

## Environment

- Python 3.10  
- Transformers (HuggingFace)  
- SentenceTransformers  
- LangChain  
- ChromaDB  
- Google Colab (T4 and A100)

---

## Ethical Considerations

This system is **not a replacement for clinical advice**. Generated responses are experimental and must be verified by licensed professionals in real-world settings.

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.  
- Reimers & Gurevych (2019). *Sentence-BERT*.  
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning with T5*.  
- Chung et al. (2022). *Scaling Instruction-Finetuned Language Models*.  
- Stanford Alpaca: [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)  
- HuggingFace: [https://huggingface.co](https://huggingface.co)  
- ChromaDB Docs: [https://docs.trychroma.com](https://docs.trychroma.com)

---

## Submitted by

- Ashok Kumar Jarugubilli  
- Neeraj Babu Vaddepalli

---
