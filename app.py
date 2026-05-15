import streamlit as st
import pdfplumber  # Better table extraction than PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import openai
from langchain.vectorstores import FAISS
import time
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from tempfile import NamedTemporaryFile
import os
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from langchain.schema import BaseRetriever

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Enhanced system prompt for maximum accuracy with language consistency and table support
SYSTEM_TEMPLATE = """You are an expert document analysis assistant with strict adherence to factual accuracy.

CRITICAL LANGUAGE RULE: You MUST respond in the SAME language as the user's question. If the user asks in English, respond in English. If they ask in Arabic, respond in Arabic. NEVER switch languages mid-conversation unless the user explicitly switches languages.

IMPORTANT: The context may contain tables in markdown format. When answering questions about tables:
- Interpret the table structure correctly
- Provide accurate data from specific rows/columns
- Compare values when asked
- Preserve numerical accuracy

CRITICAL RULES:
1. ALWAYS respond in the SAME language as the user's question (highest priority)
2. Answer ONLY using information from the provided context
3. When dealing with tables, be precise with numbers and comparisons
4. If information is not in the context, explicitly state: "I don't have that information in the provided documents" (in the user's language)
5. Never make assumptions or use external knowledge
6. Cite specific sections when answering (e.g., "According to page 3..." or "Table 2 on page 5 shows...")
7. If the question references previous conversation, maintain consistency
8. Be comprehensive but concise
9. Maintain language consistency throughout the entire response

Context from documents:
{context}"""

HUMAN_TEMPLATE = """Chat History:
{chat_history}

Question: {question}

Provide a detailed, accurate answer based solely on the context above. Remember to respond in the SAME language as the question:"""

# Create chat prompt
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


class BM25Retriever:
    """Custom BM25 retriever for keyword-based search"""
    
    def __init__(self, documents, k=4):
        self.documents = documents
        self.k = k
        
        # Tokenize documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def get_relevant_documents(self, query):
        """Retrieve documents based on BM25 scoring"""
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_indices = np.argsort(doc_scores)[-self.k:][::-1]
        
        return [self.documents[i] for i in top_indices if doc_scores[i] > 0]


def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF and convert to markdown format"""
    tables_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables from this page
                tables = page.extract_tables()
                
                if tables:
                    for table_num, table in enumerate(tables, 1):
                        if table and len(table) > 0:
                            # Convert table to markdown format
                            try:
                                # Create DataFrame for easier manipulation
                                df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                                
                                # Convert to markdown
                                markdown_table = df.to_markdown(index=False)
                                
                                # Add metadata
                                table_text = f"\n\n--- TABLE {table_num} (Page {page_num}) ---\n{markdown_table}\n--- END TABLE ---\n\n"
                                tables_text.append({
                                    'text': table_text,
                                    'page': page_num,
                                    'table_num': table_num
                                })
                            except Exception as e:
                                # Fallback: simple text representation
                                table_text = f"\n\n--- TABLE {table_num} (Page {page_num}) ---\n"
                                for row in table:
                                    table_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                                table_text += "--- END TABLE ---\n\n"
                                tables_text.append({
                                    'text': table_text,
                                    'page': page_num,
                                    'table_num': table_num
                                })
    except Exception as e:
        st.warning(f"Table extraction warning: {str(e)}")
    
    return tables_text


def get_multifile_text(multi_docs):
    """Extract text from multiple file types with enhanced table support"""
    documents = []
    
    for file in multi_docs:
        file_name = str(file.name)
        
        try:
            if file_name.endswith(".pdf"):
                # Save to temporary file for pdfplumber
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_file.flush()
                    tmp_path = tmp_file.name
                
                # Extract regular text with pdfplumber (better than PyPDF2)
                with pdfplumber.open(tmp_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        # Get page text
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            documents.append(Document(
                                page_content=page_text,
                                metadata={
                                    "source": file_name,
                                    "page": page_num,
                                    "type": "pdf_text"
                                }
                            ))
                
                # Extract tables separately
                tables = extract_tables_from_pdf(tmp_path)
                for table_data in tables:
                    documents.append(Document(
                        page_content=table_data['text'],
                        metadata={
                            "source": file_name,
                            "page": table_data['page'],
                            "type": "pdf_table",
                            "table_num": table_data['table_num']
                        }
                    ))
                
                # Clean up temp file
                os.remove(tmp_path)

            elif file_name.endswith(".txt"):
                bytes_data = file.read()
                with NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(bytes_data)
                    tmp_file.flush()

                text_reader = TextLoader(tmp_file.name, encoding='UTF-8')
                docs = text_reader.load()
                for doc in docs:
                    doc.metadata.update({"source": file_name, "type": "txt"})
                documents.extend(docs)
                os.remove(tmp_file.name)

            elif file_name.endswith((".docx", ".doc")):
                bytes_data = file.read()
                with NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                    tmp_file.write(bytes_data)
                    tmp_file.flush()

                doc_reader = Docx2txtLoader(tmp_file.name)
                docs = doc_reader.load()
                for doc in docs:
                    doc.metadata.update({"source": file_name, "type": "docx"})
                documents.extend(docs)
                os.remove(tmp_file.name)

            elif file_name.endswith((".pptx", ".ppt")):
                bytes_data = file.read()
                with NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_file:
                    tmp_file.write(bytes_data)
                    tmp_file.flush()

                ppt_reader = UnstructuredPowerPointLoader(tmp_file.name)
                docs = ppt_reader.load()
                for doc in docs:
                    doc.metadata.update({"source": file_name, "type": "pptx"})
                documents.extend(docs)
                os.remove(tmp_file.name)
                
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
            continue
            
    return documents


def get_text_chunks(documents):
    """Split documents with optimal chunking strategy for tables"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger to keep tables together
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
    
    return chunks


def get_hybrid_vectorstore(text_chunks):
    """Create vector store with hybrid retrieval capabilities"""
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai.api_key
    )
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    
    return vectorstore, text_chunks


def create_hybrid_retriever(vectorstore, text_chunks, llm):
    """Create hybrid retriever combining semantic and keyword search (no LLM reranking)"""
    
    # Semantic retriever
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,  # Reduced since we're not reranking
            "fetch_k": 20
        }
    )
    
    # BM25 keyword retriever
    bm25_retriever = BM25Retriever(text_chunks, k=6)
    
    # Combine both retrievers
    class HybridRetriever(BaseRetriever):
        semantic_ret: object
        bm25_ret: object
        
        class Config:
            arbitrary_types_allowed = True
        
        def __init__(self, semantic_ret, bm25_ret):
            super().__init__(semantic_ret=semantic_ret, bm25_ret=bm25_ret)
        
        def _get_relevant_documents(self, query: str):
            semantic_docs = self.semantic_ret.get_relevant_documents(query)
            bm25_docs = self.bm25_ret.get_relevant_documents(query)
            
            doc_dict = {}
            
            # Add semantic results
            for doc in semantic_docs:
                doc_id = f"{doc.metadata.get('source','')}_{doc.metadata.get('chunk_id','')}"
                doc_dict[doc_id] = doc
            
            # Add BM25 results
            for doc in bm25_docs:
                doc_id = f"{doc.metadata.get('source','')}_{doc.metadata.get('chunk_id','')}"
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = doc
            
            # Return top 6 combined results
            return list(doc_dict.values())[:6]
        
        async def _aget_relevant_documents(self, query: str):
            return self._get_relevant_documents(query)
    
    hybrid_ret = HybridRetriever(semantic_retriever, bm25_retriever)
    
    # Return hybrid retriever directly (no LLM reranking)
    return hybrid_ret


def get_conversation_chain(vectorstore, text_chunks):
    """Create enhanced conversation chain with hybrid retrieval"""
    
    llm = ChatOpenAI(
        openai_api_key=openai.api_key,
        model='gpt-4o-mini',
        temperature=0,
        max_tokens=1500
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',
        max_token_limit=2000
    )
    
    # Create hybrid retriever (no reranking - saves tokens!)
    hybrid_retriever = create_hybrid_retriever(vectorstore, text_chunks, llm)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": CHAT_PROMPT
        },
        return_source_documents=True,
        verbose=False,
        max_tokens_limit=3000
    )
    
    return conversation_chain


def reset_conversation():
    """Reset conversation"""
    if st.session_state.vectorstore is not None:
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vectorstore,
            st.session_state.text_chunks
        )
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.success("✅ Chat cleared! Documents still loaded.")
    else:
        st.warning("⚠️ Please upload and process documents first.")


def generate_chat():
    """Generate downloadable chat history"""
    if not st.session_state.messages:
        return
        
    with open('chatsarchive.txt', 'w', encoding='utf-8') as outfile:
        outfile.write("=" * 60 + "\n")
        outfile.write("CHAT TRANSCRIPT\n")
        outfile.write("=" * 60 + "\n\n")
        
        for msg in st.session_state.messages:
            role = msg["role"].upper()
            content = msg["content"]
            outfile.write(f'{role}:\n{content}\n\n')
            outfile.write("-" * 60 + "\n\n")


def handle_userinput(user_question):
    """Handle user input with comprehensive error handling"""
    
    with st.chat_message("user"):
        st.markdown(user_question)
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    try:
        start_time = time.time()
        
        # Get response
        response = st.session_state.conversation({'question': user_question})
        
        elapsed_time = time.time() - start_time
        
        # Display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = response['answer']
            
            # Typing effect
            words = assistant_response.split()
            for i, word in enumerate(words):
                full_response += word + " "
                if i % 3 == 0:  # Update every 3 words for smoother effect
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Performance metrics
            col1, col2 = st.columns([3, 1])
            with col2:
                st.caption(f"⏱️ {elapsed_time:.2f}s")
            
            # Source documents
            if response.get('source_documents'):
                with st.expander(f"📚 Sources ({len(response['source_documents'])} documents)"):
                    sources_seen = set()
                    for idx, doc in enumerate(response['source_documents'], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', '')
                        doc_type = doc.metadata.get('type', '')
                        table_num = doc.metadata.get('table_num', '')
                        
                        source_key = f"{source}_{page}_{table_num}"
                        
                        if source_key not in sources_seen:
                            sources_seen.add(source_key)
                            
                            # Display source info with table indicator
                            st.markdown(f"**Source {idx}:** `{source}`")
                            if doc_type == 'pdf_table':
                                st.caption(f"📊 Table {table_num} | Page {page}")
                            elif page:
                                st.caption(f"📄 Page {page} | Type: {doc_type}")
                            
                            # Show excerpt
                            excerpt = doc.page_content[:250].strip()
                            st.text(f'"{excerpt}..."')
                            st.divider()
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history = response.get('chat_history', [])
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error: {error_msg}")
        
        # Provide helpful suggestions
        if "rate limit" in error_msg.lower():
            st.info("💡 Rate limit reached. Please wait a moment and try again.")
        elif "context length" in error_msg.lower():
            st.info("💡 Context too long. Try clearing chat or asking a simpler question.")
        else:
            st.info("💡 Try rephrasing your question or check if the information is in your documents.")


def main():
    st.set_page_config(
        page_title="Advanced Multi-document Chatbot",
        page_icon="🚀",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "language_preference" not in st.session_state:
        st.session_state.language_preference = "auto"

    # Header
    st.title("Advanced Multi-document Chatbot")
    st.caption("Hybrid Search + Table Extraction + Language Consistency = Maximum Accuracy")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language preference setting
        with st.expander("🌐 Language Settings", expanded=False):
            language_mode = st.radio(
                "Language Response Mode",
                ["Auto (Match user's language)", "Force English", "Force Arabic"],
                index=0
            )
            st.caption("⚠️ 'Auto' mode matches your question language automatically")
            st.caption("💡 Recommended: Use Auto mode for best experience")
            
            if language_mode == "Force English":
                st.session_state.language_preference = "english"
            elif language_mode == "Force Arabic":
                st.session_state.language_preference = "arabic"
            else:
                st.session_state.language_preference = "auto"
        
        # Advanced settings
        with st.expander("🎛️ Advanced Settings", expanded=False):
            chunk_size = st.slider("Chunk Size", 400, 1200, 700, 50)
            chunk_overlap = st.slider("Chunk Overlap", 50, 300, 150, 50)
            retrieval_k = st.slider("Retrieval K", 4, 12, 8, 1)
            st.caption("💡 Adjust these if you're not getting good results")
        
        st.divider()
        
        # File uploader
        st.subheader("📁 Upload Documents")
        multi_docs = st.file_uploader(
            "Supported: PDF, TXT, DOC, DOCX, PPT, PPTX",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'doc', 'docx', 'pptx', 'ppt']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Process", type="primary"):
                if multi_docs:
                    with st.spinner("⚙️ Processing documents and extracting tables..."):
                        try:
                            progress_bar = st.progress(0)
                            
                            # Step 1: Extract text and tables
                            progress_bar.progress(20)
                            documents = get_multifile_text(multi_docs)
                            
                            if not documents:
                                st.error("❌ No text extracted from documents.")
                                return
                            
                            # Count tables and text
                            table_count = sum(1 for doc in documents if doc.metadata.get('type') == 'pdf_table')
                            text_count = len(documents) - table_count
                            
                            st.info(f"✓ Extracted {text_count} text sections + {table_count} tables")
                            
                            # Step 2: Create chunks
                            progress_bar.progress(40)
                            text_chunks = get_text_chunks(documents)
                            st.info(f"✓ Created {len(text_chunks)} chunks")
                            
                            # Step 3: Create vector store
                            progress_bar.progress(60)
                            vectorstore, chunks = get_hybrid_vectorstore(text_chunks)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.text_chunks = chunks
                            
                            # Step 4: Create conversation chain
                            progress_bar.progress(80)
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore, chunks
                            )
                            
                            # Step 5: Complete
                            progress_bar.progress(100)
                            st.session_state.processed_files = [f.name for f in multi_docs]
                            
                            st.success('✅ Ready to chat!')
                            time.sleep(0.5)
                            progress_bar.empty()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            progress_bar.empty()
                else:
                    st.warning("⚠️ Please upload documents first.")
        
        with col2:
            if st.button("🗑️ Clear"):
                reset_conversation()
        
        # Show stats
        if st.session_state.text_chunks:
            st.divider()
            st.subheader("📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(st.session_state.processed_files))
            with col2:
                st.metric("Chunks", len(st.session_state.text_chunks))
        
        # Show files
        if st.session_state.processed_files:
            st.divider()
            st.subheader("📑 Loaded Files")
            for file_name in st.session_state.processed_files:
                st.text(f"✓ {file_name}")
        
        # Download chat
        if st.session_state.messages:
            st.divider()
            generate_chat()
            with open('chatsarchive.txt', 'r', encoding='utf-8') as f:
                st.download_button(
                    "💾 Download Chat",
                    f,
                    file_name="chat_history.txt"
                )
        
        # Info section
        st.divider()
        st.caption("🚀 **Features:**")
        st.caption("• Language consistency (Auto mode)")
        st.caption("• Accurate table extraction (pdfplumber)")
        st.caption("• Hybrid semantic + keyword search")
        st.caption("• Smart chunking with overlap")
        st.caption("• Source citation")
        st.caption("• Context-aware conversations")

    # Main chat area
    if st.session_state.conversation is None:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 Upload documents to begin")
            # st.markdown("---")
            # st.subheader("✨ What makes this chatbot advanced?")
            # st.markdown("""
            # - **Language Consistency**: Matches your question language automatically
            # - **Table Extraction**: Accurate table reading with pdfplumber
            # - **Hybrid Search**: Combines semantic understanding with keyword matching
            # - **Smart Chunking**: Optimized text splitting for better context
            # - **Source Tracking**: See exactly where answers come from
            # - **Context Memory**: Maintains conversation history
            # """)
            
            # st.markdown("---")
            st.subheader("💬 Example Questions:")
            st.markdown("""
            **English:**
            - *"What is the main topic of this document?"*
            - *"Summarize the key findings on page 5"*
            - *"What are the values in the sales table?"*
            - *"Compare Q1 and Q2 revenue from the table"*
            
            **العربية:**
            - *"ما هو الموضوع الرئيسي لهذه الوثيقة؟"*
            - *"لخص النتائج الرئيسية في الصفحة 5"*
            - *"ما هي القيم في جدول المبيعات؟"*
            """)
    else:
        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if user_question := st.chat_input("💬 Ask anything about your documents..."):
            handle_userinput(user_question)


if __name__ == '__main__':
    main()
