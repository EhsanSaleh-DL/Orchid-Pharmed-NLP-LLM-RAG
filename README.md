# Orchid-Pharmed-NLP-LLM-RAG
LLM-Based Patient Review Analysis &amp; RAG Question Answering System
# LLM-Based Patient Review Analysis & RAG Question Answering System

This repository contains the Python implementations developed for the second-stage technical interview task at **Orchid Pharmed**.

The project focuses on two practical applications of Large Language Models (LLMs):

1. **Patient Review Sentiment Analysis and Insight Extraction**
2. **Retrieval-Augmented Generation (RAG) based Question Answering**

The project was implemented primarily in Python and Google Colab.

---

## Project Overview

The task consisted of two main components.

### Task 1 — Sentiment Analysis and Insight Extraction

The objective was to build a sentiment analysis system for hospital patient reviews using a Large Language Model (LLM), while also extracting meaningful insights from the reviews.

The analysis includes:

* Sentiment labeling
* Three-class sentiment classification
* Positive, Negative, and Neutral categories
* Frequent keyword / n-gram analysis
* High-level topic extraction
* Aspect-level analysis of positive and negative feedback

The original task specification explicitly required the use of an LLM and meaningful insight extraction from patient reviews.

---

### Task 2 — Question & Answering System

The second task was to develop a Question & Answering system for a PDF document.

The main objective was to demonstrate a **Retrieval-Augmented Generation (RAG)** pipeline using a language model and vector similarity search.

The implemented pipeline consists of:

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Text Embeddings
     ↓
FAISS Vector Store
     ↓
Similarity Retrieval
     ↓
LLM
     ↓
Context-Grounded Answer
```

---

# Task 1 — Patient Review Analysis

## 1. Data Preparation and Sentiment Labeling

During the initial inspection of the hospital review dataset, it was observed that the dataset did not explicitly contain a Neutral class, although Neutral sentiment was required by the task.

Reviews with a rating of 3 were therefore considered candidates for the Neutral class.

In addition, the textual content and rating information were examined to identify potential inconsistencies between the existing labels and the ratings.

A rule-based sentiment labeling stage was implemented using **VADER (Valence Aware Dictionary and sEntiment Reasoner)** together with the numerical rating.

The resulting sentiment classes were encoded numerically as:

```text
Negative → 0
Positive → 1
Neutral  → 2
```

The resulting dataset contained:

* Positive: 646 reviews
* Negative: 304 reviews
* Neutral: 46 reviews

The processed labels were saved to a new CSV file for subsequent modeling and analysis.

---

## 2. LLM-Based Sentiment Classification

A transformer-based model was then used for three-class sentiment classification.

The implementation uses:

* Hugging Face Transformers
* PyTorch
* Hugging Face Datasets
* DistilBERT
* Scikit-learn

The dataset was divided into training, validation, and test subsets using stratified splitting so that the class distribution was preserved across the subsets.

Because the Neutral class was substantially smaller than the other classes, class weighting was incorporated into the training loss.

### Model

The sentiment classifier is based on **DistilBERT**, a lightweight transformer architecture derived from BERT.

The text is tokenized using the DistilBERT tokenizer and passed through the transformer model. The resulting representation is then used for three-class classification.

The training procedure includes:

* Stratified train/validation/test splitting
* DistilBERT tokenization
* Weighted cross-entropy loss
* Class weighting
* Early stopping
* Validation monitoring
* Final evaluation on the unseen test set

Evaluation includes:

* Accuracy
* Macro F1-score
* Classification report

The best-performing model is saved and subsequently used for inference through a pipeline.

---

# Task 1 — Insight Extraction

The second part of Task 1 focuses on extracting meaningful insights from patient feedback.

Several complementary NLP approaches were implemented.

---

## 3. Frequent N-gram Analysis

Frequent phrases were extracted using `CountVectorizer` from Scikit-learn.

The implementation uses N-gram analysis, particularly **bigrams**, after removing English stop words.

The main functions include:

```text
ngrams_top_get
ngrams_plot
```

The analysis is performed separately for positive and negative reviews.

The resulting frequency plots provide an overview of the phrases that occur most frequently in each sentiment class.

This can help identify recurring issues, strengths, and common themes in patient feedback.

---

## 4. High-Level Topic Modeling

High-level themes were extracted using **BERTopic**.

The analysis is performed separately for different sentiment classes.

The pipeline uses:

```text
Text
 ↓
Semantic Representation
 ↓
UMAP Dimensionality Reduction
 ↓
Clustering
 ↓
BERTopic
 ↓
Topics
```

A fixed random state is used for UMAP to improve reproducibility.

The resulting topics provide a higher-level view of the main subjects discussed in patient reviews.

Examples of identified themes include:

* Doctors and doctor-patient interaction
* Nursing and staff care
* Emergency department experiences
* Cleanliness and facilities
* General hospital experience
* Service and support

---

## 5. Aspect Extraction from Reviews

For negative reviews, adjective-noun pairs are extracted using the English spaCy language model.

Examples include patterns such as:

```text
dirty room
rude staff
```

The most frequent extracted phrases are then visualized to identify recurring complaints.

The same approach is also applied to positive reviews to identify aspects associated with positive experiences.

---

## 6. Zero-Shot Aspect Classification

A zero-shot classification approach is also used to identify business/service aspects discussed in patient feedback.

The implementation uses:

```text
facebook/bart-large-mnli
```

Example aspects include:

* Doctors
* Nurses
* Staff behavior
* Waiting time
* Cleanliness
* Facilities
* Quality of treatment

Each review is assigned to one of the predefined aspects without requiring task-specific supervised training.

The resulting aspect frequencies are visualized to identify the areas associated with positive and negative patient experiences.

---

# Task 2 — Question & Answering System

## Retrieval-Augmented Generation (RAG)

The second task implements a document-based Question & Answering system using Retrieval-Augmented Generation.

The system uses an academic paper as the source document.

### Pipeline

```text
PDF
 ↓
PyPDF2
 ↓
Extracted Text
 ↓
Text Chunks
 ↓
OpenAI Embeddings
 ↓
FAISS
 ↓
Similarity Search
 ↓
Relevant Context
 ↓
GPT-4o
 ↓
Answer
```

---

## 1. PDF Text Extraction

The PDF is loaded using `PyPDF2`.

The text from each page is extracted and combined into a single text document.

The extracted content is temporarily saved as:

```text
sample.txt
```

---

## 2. Text Chunking

The extracted document is divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

Configuration:

```text
chunk_size = 1000
chunk_overlap = 100
```

Chunking makes it possible to retrieve only the relevant portions of the document instead of passing the complete document to the language model.

---

## 3. Embeddings

Each text chunk is converted into a numerical vector representation using:

```text
text-embedding-3-small
```

These embeddings represent the semantic content of the text and enable similarity-based retrieval.

---

## 4. FAISS Vector Database

The generated embeddings are stored in a **FAISS** vector store.

FAISS enables efficient similarity search over the embedded document chunks.

The vector store can also be saved locally:

```python
vectorstore.save_local("faiss_index")
```

and loaded again later without rebuilding the embeddings.

---

## 5. Similarity Retrieval

A user question is converted into a semantic search query.

The system retrieves the most relevant document chunks using similarity search.

The retriever is configured to return the top relevant chunks.

---

## 6. RAG Question Answering

The retrieved chunks are passed to the language model as context.

The prompt explicitly instructs the model to answer **only using the retrieved context**.

If the answer cannot be found in the provided context, the system is instructed to return:

```text
I don't know.
```

This reduces the risk of generating answers that are unsupported by the source document.

---

## Example

The system was tested with the question:

> What are CNN models used in this paper?

The RAG system correctly retrieved the relevant section and generated the answer:

> EfficientNet-B0, EfficientNet-B1, EfficientNet-B2, Inception-v3, and Xception.

---

# Repository Structure

```text
orchid-pharmed-nlp-llm-rag/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Task_1/
│   ├── Part_1/
│   └── Part_2/
│
├── Task_2/
│   ├── Question_and_Answering_System.ipynb
│   └── README.md
│
└── results/
    ├── sentiment/
    ├── insights/
    └── rag/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/orchid-pharmed-nlp-llm-rag.git
cd orchid-pharmed-nlp-llm-rag
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# OpenAI API Key

The project uses OpenAI models for embeddings and language-model inference.

For security reasons, API keys are **not included in this repository**.

Before running the RAG notebook, configure your OpenAI API key as an environment variable:

```python
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
```

Never place a personal API key directly inside the notebook or source code.

---

# Running the Notebooks

The implementations were originally developed in Google Colab.

The notebooks can be opened and executed using Google Colab after configuring the required data and API credentials.

---

# Results

The detailed analysis and results of the project are documented in the accompanying project report.

Selected result visualizations are also included in the `results/` directory to provide a quick overview of the analyses.

The repository is intended primarily to demonstrate the implementation and methodology. The accompanying report provides additional discussion and interpretation of the results.

---

# Technologies

The project uses several tools and libraries from the modern NLP and LLM ecosystem:

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT
* Scikit-learn
* NLTK / VADER
* BERTopic
* UMAP
* spaCy
* Hugging Face Zero-Shot Classification
* LangChain
* OpenAI API
* FAISS
* PyPDF2
* Google Colab

---

# Disclaimer

This repository was developed as part of a technical interview task.

The code is provided to demonstrate the implemented approaches and does not represent a production-ready clinical decision-support system.

Patient data and other potentially sensitive source data are not included in this repository.

---

# Author

**Ehsan Saleh**

M.Sc. Biomedical Engineering – Bioelectric
Iran University of Science and Technology

Research interests include:

* Machine Learning & Deep Learning
* Natural Language Processing
* Large Language Models
* Retrieval-Augmented Generation
* Medical AI
* Medical Image and Signal Processing

For further information regarding the implementation, please feel free to contact the author.
