import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np

# Set paths
RAW_DATA_DIR = r"C:\Users\saipr\Desktop\Untitled Folder 1\chatbot_theme_identifier\backend\data"
EXTRACTED_DIR = os.path.join(RAW_DATA_DIR, "extracted")
FAISS_INDEX_PATH = r"C:\Users\saipr\Desktop\Untitled Folder 1\chatbot_theme_identifier\backend\faiss_index"

st.set_page_config(page_title="🧠 ML Research Navigator", layout="wide")
st.markdown("# 🧠 ML Research Navigator")
st.markdown("Use this assistant to explore Machine Learning research reports through intelligent search.")

# Ask a question
st.markdown("""
<style>
.big-font {
    font-size: 22px !important;
    font-weight: bold;
    color: #004d99;
}
</style>
<div class="big-font">🔍 Ask a Question:</div>
""", unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g., What is CNN used for?")

# Sidebar document list
with st.sidebar:
    st.header("📂 Uploaded Research Files")
    st.subheader("📄 PDF Reports")
    for f in sorted(os.listdir(RAW_DATA_DIR)):
        if f.endswith(".pdf"):
            st.markdown(f"- `{f}`")
    st.subheader("🖼️ Scanned Image Reports")
    for f in sorted(os.listdir(RAW_DATA_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            st.markdown(f"- `{f}`")

@st.cache_resource
def load_faiss():
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return FAISS.load_local(FAISS_INDEX_PATH, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    else:
        docs = []
        for filename in os.listdir(EXTRACTED_DIR):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(EXTRACTED_DIR, filename), encoding="utf-8")
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = filename
                docs.extend(loaded_docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
        db.save_local(FAISS_INDEX_PATH)
        return db

# Load index
db = load_faiss()
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 35})
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=llm_pipeline)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

if query:
    with st.spinner("Thinking..."):
        result = qa.invoke({"query": query})
        answer = result['result']
        source_docs = result['source_documents']

        st.success("✅ Answer:")
        st.markdown(f"**{answer.strip()}**")

        # Split into PDF and image results
        pdf_matches, img_matches = [], []
        for doc in source_docs:
            source = doc.metadata.get("source", "").lower()
            if "img_doc_" in source:
                img_matches.append(doc)
            else:
                pdf_matches.append(doc)

        final_matches = pdf_matches[:3] + img_matches[:3]

        if final_matches:
            st.markdown("### ✅ Top Matching Results")
            for i, doc in enumerate(final_matches):
                source = doc.metadata.get("source", "unknown")
                doc_type = "🖼️ Image OCR" if "img_doc_" in source else "📄 PDF Document"
                content_preview = doc.page_content.strip().replace('\n', ' ')
                line_summary = content_preview.split('.')[:2]
                summary_line = '. '.join(line_summary).strip()

                if "img_doc_" in source:
                    loc = f"📃 **Line**: {doc.metadata.get('line', 'N/A')}"
                else:
                    loc = f"📃 **Page**: {doc.metadata.get('page', 'N/A')}<br>📑 **Paragraph**: {doc.metadata.get('paragraph', 'N/A')}"

                st.markdown(f"""
📌 **Match {i+1}**

📁 **Source**: `{source}`<br>
{loc}<br>
🧾 **Type**: {doc_type}
```text
{summary_line}...
```
""", unsafe_allow_html=True)
        else:
            st.warning("No matching documents found.")

# Phase 2: Theme Extraction from ALL docs
st.markdown("---")
st.markdown("### 🧠 Cross-Document Theme Identification")

# Get all chunks
all_chunks = db.similarity_search("", k=1000)
texts = [d.page_content for d in all_chunks]
sources = [d.metadata.get("source", "") for d in all_chunks]

# TF-IDF + KMeans
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X = vectorizer.fit_transform(texts)
true_k = min(7, len(texts)//3)
kmeans = KMeans(n_clusters=true_k, random_state=42).fit(X)

clusters = defaultdict(list)
for idx, label in enumerate(kmeans.labels_):
    clusters[label].append((texts[idx], sources[idx]))

# Summarize themes
for i, entries in clusters.items():
    docs_in_cluster = list(set([src for _, src in entries]))
    if len(docs_in_cluster) < 2:
        continue

    top_text = ' '.join([text for text, _ in entries[:5]])[:1500]
    summary_prompt = f"Summarize the following into a clear research theme in 5 lines:\n\n{top_text}"
    summary = llm_pipeline(summary_prompt)[0]['generated_text'].strip()

    st.markdown(f"""
🧩 **Theme {i+1}**

**Summary**: {summary}

**Appears in**: {', '.join(sorted(set(docs_in_cluster)))}
""")
