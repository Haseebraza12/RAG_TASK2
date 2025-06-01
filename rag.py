import os
import sys
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import requests
from typing import Any, List, Optional, Mapping
import warnings

# === FIX TORCH ERROR ===
try:
    import torch
    if hasattr(sys.modules['torch.classes'], '__path__'):
        sys.modules['torch.classes'].__path__ = []
except Exception:
    pass

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- API Key ---
GROQ_API_KEY = "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"  # Replace with your actual API key

class GroqLLM(LLM):
    model_name: str = "llama-3.3-70b-versatile"  # Updated model name
    temperature: float = 0.1
    max_tokens: int = 1000
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided documents. If you cannot answer from the context, say: 'I can only answer questions related to the provided document.'"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()  # Raise exception for non-200 status codes
            
            return response.json()["choices"][0]["message"]["content"]
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request Error: {str(e)}"
            st.error(error_msg)
            return f"I apologize, but I encountered a network error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            st.error(error_msg)
            return f"I apologize, but I encountered an unexpected error: {error_msg}"

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

def setup_page_config():
    st.set_page_config(
        page_title="RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            border: 1px solid #e0e0e0;
        }
        .user { background-color: #E8F0FE; }
        .assistant { background-color: #F8F9FA; }
        .message-content {
            margin-top: 0.5rem;
            white-space: pre-wrap;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def process_document(uploaded_file):
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "uploaded_file.pdf")
        
        with open(temp_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n",
            length_function=len
        )
        split_documents = text_splitter.split_documents(documents)
        
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(split_documents, embeddings)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return vector_store, documents
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, None

def display_chat_message(role: str, content: str):
    display_role = "You" if role == "user" else "Assistant"
    st.markdown(
        f"""
        <div class="chat-message {role}">
            <div><strong>{display_role}</strong></div>
            <div class="message-content">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.conversation_history = []
        st.session_state.vector_store = None
        st.session_state.documents = None

def main():
    initialize_session_state()
    setup_page_config()
    
    st.title("🤖 Document Q&A System")
    st.markdown("### Upload a document and start asking questions")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
        if st.button("Clear Chat History"):
            st.session_state.conversation_history = []
            st.rerun()
    
    if uploaded_file:
        if st.session_state.get('vector_store') is None:
            with st.spinner("Processing document..."):
                vector_store, documents = process_document(uploaded_file)
                if vector_store and documents:
                    st.session_state.vector_store = vector_store
                    st.session_state.documents = documents
        
        if st.session_state.vector_store:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )
            
            llm = GroqLLM()
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
            
            user_query = st.text_input(
                "Ask your question:",
                placeholder="What would you like to know about the document?",
                key="user_input"
            )
            
            if user_query:
                with st.spinner("Processing your question..."):
                    try:
                        result = qa_chain({"question": user_query})
                        answer = result['answer']
                        
                        st.session_state.conversation_history.extend([
                            {"role": "user", "content": user_query},
                            {"role": "assistant", "content": answer}
                        ])
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if st.session_state.conversation_history:
                st.markdown("### Conversation History")
                for message in st.session_state.conversation_history:
                    display_chat_message(message["role"], message["content"])
    else:
        st.info("👆 Please upload a PDF file to begin")

if __name__ == "__main__":
    main()