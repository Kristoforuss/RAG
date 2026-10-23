import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

# Impor komponen LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.chains import ConversationBufferMemory

# Impor komponen DeepSeek (menggunakan wrapper OpenAI)
from langchain_openai import ChatOpenAI

# Impor komponen Pinecone & HuggingFace (E5 Model)
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec

# --- Konfigurasi Halaman & Variabel Lingkungan ---
load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant (Pinecone + E5 + DeepSeek)",
    page_icon="🤖",
    layout="wide"
)

INDEX_NAME = "pdfchatbotdemo"

# --- Fungsi Pemrosesan RAG ---

def process_pdf(uploaded_files):
    """
    Menyimpan file PDF yang diunggah ke temporary file dan mengekstrak teksnya.
    """
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filepath = temp_file.name

        loader = PyPDFLoader(temp_filepath)
        documents.extend(loader.load())
        os.remove(temp_filepath)

    return documents

def get_text_chunks(documents):
    """
    Memecah dokumen menjadi potongan (chunks) kecil.
    Tambahkan prefix 'passage: ' sesuai anjuran model E5 untuk dokumen yang disimpan.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # E5 merekomendasikan prefix untuk data korpus
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"

    return chunks

def get_embeddings_model():
    """
    Menginisialisasi Model Embedding multilingual-e5-large
    """
    return PineconeEmbeddings(model="multilingual-e5-large")

def setup_pinecone_and_embed(chunks):
    """
    Menginisialisasi index Pinecone (jika belum ada) dan melakukan upsert.
    """
    embeddings = get_embeddings_model()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in pc.list_indexes().names():
        st.info(f"Membuat index Pinecone '{INDEX_NAME}'. Ini butuh waktu sekitar 10-30 detik...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024, # Dimensi output e5-large
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    return vector_store

def get_existing_vector_store():
    """
    Mengambil koneksi ke vector store yang sudah ada.
    """
    embeddings = get_embeddings_model()
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

def get_conversational_chain(vector_store):
    """
    Membangun rantai percakapan (chain) menggunakan DeepSeek LLM.
    """
    # Menyelaraskan API Key dengan pengecekan di sidebar (DEEPSEEK_API_KEY)
    llm = ChatOpenAI(
        api_key=os.getenv("TOKEHARBOR_API_KEY"),
        base_url="https://tokenharbor.ai/v1",
        model="deepseek-v4.1-flash:free",
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True
    )

    return conversation_chain

# --- Antarmuka Utama (Streamlit) ---

def main():
    st.title("🤖 Chat PDF dengan DeepSeek AI")
    st.markdown("Didukung oleh **Pinecone**, model **E5 Multilingual**, dan LLM **DeepSeek**.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.header("📂 Kelola Dokumen")
        pdf_docs = st.file_uploader(
            "Unggah file PDF baru ke Pinecone",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Proses & Embed"):
            if not pdf_docs:
                st.warning("Silakan unggah dokumen terlebih dahulu.")
                return

            if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("PINECONE_API_KEY"):
                st.error("Pastikan DEEPSEEK_API_KEY dan PINECONE_API_KEY ada di file .env")
                return

            with st.spinner("Mengekstrak dan upsert ke Pinecone..."):
                raw_documents = process_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_documents)
                vector_store = setup_pinecone_and_embed(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Dokumen berhasil disisipkan! Silakan mulai bertanya.")

        st.divider()
        st.markdown("### Atau Gunakan Index Lama")
        if st.button("Muat Database Pinecone"):
            if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("PINECONE_API_KEY"):
                st.error("Pastikan DEEPSEEK_API_KEY dan PINECONE_API_KEY ada di file .env")
                return

            with st.spinner("Menghubungkan..."):
                vector_store = get_existing_vector_store()
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Terhubung ke Pinecone! Anda bisa langsung bertanya.")

    # Area Chat Utama - Menggunakan tipe pesan bawaan LangChain
    for message in st.session_state.chat_history:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(message.content)

    user_question = st.chat_input("Tanyakan sesuatu tentang dokumen Anda...")

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Pilih 'Proses & Embed' atau 'Muat Database Pinecone' di sidebar terlebih dahulu.")
            return

        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("DeepSeek sedang menganalisis dokumen Anda..."):
            e5_formatted_query = f"query: {user_question}"
            response = st.session_state.conversation.invoke({"question": e5_formatted_query})
            st.session_state.chat_history = response['chat_history']

            with st.chat_message("assistant"):
                st.write(response['answer'])

if __name__ == "__main__":
    main()
