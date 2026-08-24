\# 📚 Research RAG Assistant



An end-to-end Retrieval-Augmented Generation (RAG) system for asking questions about research papers using hybrid information retrieval and Gemini.



The system combines \*\*semantic search, keyword search, and hybrid ranking\*\* to retrieve relevant passages from research papers and uses \*\*Google Gemini\*\* to generate grounded answers with source references.



\---



\## 🎯 Project Overview



Research papers often contain large amounts of technical information, making it difficult to quickly locate specific concepts, definitions, and comparisons.



This project provides a research assistant that allows users to:



\- Upload research papers in PDF format

\- Process and chunk the documents

\- Retrieve relevant passages using semantic and keyword search

\- Combine retrieval results using Reciprocal Rank Fusion (RRF)

\- Generate answers using Google Gemini

\- Display the papers, pages, and chunks used to generate each answer



The system is designed to reduce hallucination by instructing the LLM to answer only from the retrieved document context.



\---



\## ✨ Key Features



\- 📄 \*\*PDF ingestion\*\* using PyMuPDF

\- ✂️ \*\*Overlapping text chunking\*\*

\- 🧠 \*\*Semantic embeddings\*\* using BGE

\- 🔎 \*\*FAISS vector search\*\*

\- 🔤 \*\*BM25 keyword retrieval\*\*

\- 🔀 \*\*Reciprocal Rank Fusion (RRF)\*\* for hybrid ranking

\- 🤖 \*\*Gemini-powered answer generation\*\*

\- 📚 \*\*Source and page-level references\*\*

\- 📊 \*\*Retrieval evaluation\*\*

\- 🌐 \*\*Interactive Streamlit interface\*\*

\- 🔄 \*\*Dynamic index rebuilding after uploading documents\*\*

\- 🛡️ \*\*Context-grounded responses\*\*



\---



\# 🛠️ Technology Stack



| Component | Technology |

|---|---|

| Programming Language | Python |

| User Interface | Streamlit |

| PDF Processing | PyMuPDF |

| Embedding Model | BAAI/bge-small-en-v1.5 |

| Semantic Retrieval | FAISS |

| Keyword Retrieval | BM25 |

| Hybrid Ranking | Reciprocal Rank Fusion (RRF) |

| Large Language Model | Google Gemini |

| Numerical Computing | NumPy |

| Evaluation | Recall@5, MRR |



\---



\# 🏗️ System Architecture



```text

&#x20;                   Research Papers

&#x20;                         │

&#x20;                         ▼

&#x20;                   PDF Processing

&#x20;                      PyMuPDF

&#x20;                         │

&#x20;                         ▼

&#x20;                  Text Extraction

&#x20;                         │

&#x20;                         ▼

&#x20;                Chunking + Overlap

&#x20;                         │

&#x20;                         ▼

&#x20;                 BGE Embeddings

&#x20;                         │

&#x20;                         ▼

&#x20;            ┌─────────────────────────┐

&#x20;            │                         │

&#x20;            ▼                         ▼

&#x20;       FAISS Search              BM25 Search

&#x20;     Semantic Retrieval       Keyword Retrieval

&#x20;            │                         │

&#x20;            └───────────┬─────────────┘

&#x20;                        ▼

&#x20;               Reciprocal Rank

&#x20;                   Fusion

&#x20;                      RRF

&#x20;                        │

&#x20;                        ▼

&#x20;                 Top-K Chunks

&#x20;                        │

&#x20;                        ▼

&#x20;                 Context Builder

&#x20;                        │

&#x20;                        ▼

&#x20;                     Gemini

&#x20;                        │

&#x20;                        ▼

&#x20;                Grounded Answer

&#x20;                        │

&#x20;                        ▼

&#x20;               Answer + Sources



🔍 How the RAG Pipeline Works



1.PDF Ingestion



Research papers are uploaded through the Streamlit interface and stored locally.



PyMuPDF extracts text from each page while preserving page information.



2\. Text Chunking



Extracted text is divided into smaller overlapping chunks.



The current configuration uses:



Chunk size: 350 words

Overlap: 50 words



Page information and document metadata are preserved with every chunk.



Each chunk contains:



chunk\_id

paper

page

chunk\_number

text



3\. Embedding Generation



Each text chunk is converted into a dense vector representation using:



BAAI/bge-small-en-v1.5



These embeddings allow the system to perform semantic similarity search.



4\. FAISS Semantic Retrieval



FAISS is used to search for chunks that are semantically similar to the user's question.



This helps retrieve relevant information even when the question does not use exactly the same words as the research paper.



5\. BM25 Keyword Retrieval



BM25 performs keyword-based retrieval over the document chunks.



This is particularly useful for technical research questions containing specific terms such as:



LoRA

QLoRA

NF4

quantization

fine-tuning



6\. Hybrid Retrieval with RRF



The FAISS and BM25 results are combined using Reciprocal Rank Fusion (RRF).



The RRF score is calculated as:



RRF(d) = Σ 1 / (k + rank(d))



where:



k = 60



This combines semantic and lexical retrieval into a single ranking.



The top 5 chunks are then selected as the context for the LLM.



7\. Context Construction



The selected chunks are combined into a structured context containing:



Paper name

Page number

Chunk text



This context is passed to Gemini along with the user's question.



8\. Gemini Answer Generation



Google Gemini receives the retrieved context and the user's question.



The model is instructed to:



* Answer only from the provided context
* Avoid inventing information
* Clearly answer the question
* State when the answer cannot be found in the documents



This creates a grounded RAG response instead of asking the LLM to answer entirely from its pretrained knowledge.



9\. Source Attribution



The application displays the research papers and page numbers associated with the retrieved chunks.



Example:



📄 QLoRA.pdf

📑 Page 5

🔹 Chunk 3



This allows users to trace generated answers back to the original research papers.



📊 Evaluation



The retrieval system was evaluated using standard information-retrieval metrics.



Current Results

Metric	       Result

Recall@5	100%

MRR	        0.800





Recall@5



Recall@5 measures whether a relevant document chunk appears within the top 5 retrieved results.



The current evaluation achieved:



Recall@5 = 100%





Mean Reciprocal Rank (MRR)



MRR measures how highly the first relevant result appears in the ranked retrieval results.



The current evaluation achieved:



MRR = 0.800



These results indicate that the hybrid retrieval pipeline is successfully retrieving relevant research-paper passages near the top of the search results.



📂 Project Structure



Research-RAG/

│

├── app.py

├── llm.py

├── build\_index.py

├── evaluate.py

├── requirements.txt

├── README.md

│

├── data/

│   ├── raw\_papers/

│   ├── processed/

│   ├── chunks/

│   └── vector\_db/

│

└── evaluation\_results.json



Main Files



app.py



Streamlit application providing the user interface for document upload, indexing, question answering, and source display.



build\_index.py



Processes PDFs, creates chunks, generates embeddings, and builds the FAISS vector index.



llm.py



Performs hybrid retrieval using FAISS + BM25 + RRF and sends the retrieved context to Gemini.



evaluate.py



Runs retrieval evaluation and reports metrics such as Recall@5 and MRR.



requirements.txt



Contains the Python dependencies required to run the project.





⚙️ Installation



1\. Clone the repository



git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd Research-RAG



2\. Create a virtual environment



Windows:

python -m venv venv



Activate it:

.\\venv\\Scripts\\Activate.ps1



3\. Install dependencies

pip install -r requirements.txt





🔑 Gemini API Configuration



The application requires a Google Gemini API key.



For security, do not hardcode the API key inside Python source files or commit it to GitHub.



Set the API key using an environment variable or Streamlit secrets.



Example environment variable:



GEMINI\_API\_KEY=your\_api\_key\_here





▶️ Running the Application



Start the Streamlit application:



python -m streamlit run app.py



Then open:



http://localhost:8501





💬 Example Questions



You can ask questions such as:



1. What is LoRA?
2. What is QLoRA?
3. How does QLoRA reduce memory usage compared with regular fine-tuning?
4. Why is NF4 used in QLoRA?
5. What are the advantages of parameter-efficient fine-tuning?



🔄 Updating the Knowledge Base



New research papers can be uploaded through the Streamlit interface.



After uploading documents:



1. Upload the PDF files
2. Select Build / Update RAG Index
3. The documents are processed
4. New chunks and embeddings are generated
5. The FAISS index is rebuilt
6. Questions can then be asked about the updated document collection



🧠 Why Hybrid Retrieval?





Using only one retrieval technique can miss relevant information.



FAISS



Provides semantic retrieval and can identify conceptually similar passages.



BM25



Provides lexical retrieval and performs well for exact technical terminology.



RRF



Combines both rankings to create a more robust retrieval system.



Therefore:



Semantic Search

&#x20;      +

Keyword Search

&#x20;      ↓

&#x20;     RRF

&#x20;      ↓

Better Hybrid Ranking



🛡️ Hallucination Control





The application uses retrieved document context to ground Gemini's responses.



The generation prompt instructs the model to:



Answer using only the provided context.

Do not invent information.

If the answer cannot be found, say that it could not be found in the provided documents.



This helps reduce unsupported responses and makes the system more suitable for research-paper question answering.





🚧 Future Improvements



Potential improvements include:



* Cross-encoder reranking
* Conversational memory
* Streaming LLM responses
* Improved citation highlighting
* Larger evaluation datasets
* Multi-document comparison
* Table and figure extraction
* OCR support for scanned papers
* Cloud deployment
* User authentication
* Per-user document collections



📌 Project Highlights





* Built an end-to-end RAG pipeline for research papers
* Implemented hybrid semantic + lexical retrieval
* Combined FAISS and BM25 using Reciprocal Rank Fusion
* Integrated Gemini for grounded answer generation
* Implemented source-level attribution
* Built an interactive Streamlit interface
* Evaluated retrieval using Recall@5 and MRR



👩‍💻 Author



Sneha Agarwal



Built as an end-to-end research-focused Retrieval-Augmented Generation project.

