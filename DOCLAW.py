import os
import io
import tempfile
import logging
import hashlib
import pickle
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import streamlit as st
import numpy as np
import pdf2image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("./cache")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "claude-3-opus-20240229"
MAX_PDF_SIZE_MB = 100  # Maximum PDF size in MB
MAX_WORKERS = 4  # Maximum number of concurrent workers for parallel processing

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(exist_ok=True)

# Cache decorator for expensive functions
def cached(func):
    def wrapper(*args, **kwargs):
        # Create a unique key based on function name and arguments
        key_parts = [func.__name__]
        # Add serializable args to key
        for arg in args:
            if hasattr(arg, 'name'):  # Handle file-like objects
                key_parts.append(arg.name)
            else:
                try:
                    key_parts.append(str(arg))
                except:
                    pass
        # Generate key
        key = hashlib.md5(str(key_parts).encode()).hexdigest()
        cache_file = CACHE_DIR / f"{key}.pkl"
        
        # If cache exists and is valid, load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    logger.info(f"Loading cached result for {func.__name__}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Otherwise compute and cache result
        result = func(*args, **kwargs)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Caching failed: {e}")
        
        return result
    return wrapper

def file_hash(file_data: bytes) -> str:
    """Generate a hash for file data to use as cache key"""
    return hashlib.md5(file_data).hexdigest()

def process_page(page, page_num: int, total_pages: int, progress_bar) -> str:
    """Process a single PDF page, extracting text or performing OCR if needed"""
    try:
        # First try direct text extraction
        page_text = page.get_text()
        
        # If minimal text extracted, try OCR
        if len(page_text.strip()) < 10:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # Increase resolution for better OCR
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(img)
        
        # Update progress
        progress_value = (page_num + 1) / total_pages
        progress_bar.progress(progress_value, text=f"Processing page {page_num + 1}/{total_pages}")
        
        return page_text
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        return f"[Error extracting page {page_num}]"

def extract_text_from_pdf(pdf_file, progress_bar=None) -> str:
    """Extract text from PDF with optimizations for large documents"""
    # Rewind file pointer
    pdf_file.seek(0)
    
    # Create a hash of the file for caching
    file_bytes = pdf_file.read()
    file_key = file_hash(file_bytes)
    cache_file = CACHE_DIR / f"pdf_text_{file_key}.txt"
    
    # Check cache
    if cache_file.exists():
        logger.info(f"Loading extracted text from cache")
        return cache_file.read_text(encoding='utf-8')
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    text = ""
    doc = None
    try:
        # Open the PDF with PyMuPDF
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        
        # Show warning for very large PDFs
        if total_pages > 100:
            st.warning(f"Large PDF detected ({total_pages} pages). Processing may take several minutes.")
        
        # Create progress bar if not provided
        if progress_bar is None:
            progress_bar = st.progress(0, text="Starting PDF processing...")
        
        # Process pages
        all_text = []
        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = process_page(page, page_num, total_pages, progress_bar)
            all_text.append(page_text)
        
        text = "\n\n".join(all_text)
        
        # Cache the result
        cache_file.write_text(text, encoding='utf-8')
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        st.error(f"Error processing PDF: {e}")
    finally:
        if doc is not None:
            doc.close()
        os.unlink(temp_path)
    
    return text

@cached
def extract_text_from_image(image_file) -> str:
    """Extract text from an image using Tesseract OCR with caching"""
    # Rewind file pointer
    image_file.seek(0)
    
    try:
        image = Image.open(image_file)
        
        # Optimize image for OCR
        # Resize large images to improve OCR speed
        max_size = 3000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to grayscale to improve OCR accuracy
        if image.mode != 'L':
            image = image.convert('L')
        
        # Improve contrast
        import numpy as np
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Use advanced OCR config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        st.error(f"Error processing image: {e}")
        return ""

@cached
def chunk_text(text: str) -> List[str]:
    """Split text into manageable chunks with optimized parameters"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        st.error(f"Error preparing document chunks: {e}")
        return []

@st.cache_resource
def load_embeddings():
    """Load embedding model with caching"""
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        st.error(f"Failed to load embedding model: {e}")
        return None

@cached
def create_faiss_index(chunks: List[str]) -> Any:
    """Create a FAISS index from text chunks with progress tracking"""
    if not chunks:
        raise ValueError("No text chunks provided")
    
    try:
        embeddings = load_embeddings()
        if embeddings is None:
            raise ValueError("Failed to initialize embeddings")
        
        with st.spinner("Creating vector embeddings for document chunks..."):
            faiss_index = FAISS.from_texts(chunks, embeddings)
        return faiss_index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        st.error(f"Error creating search index: {e}")
        return None

def query_llm(faiss_index: Any, query: str, temperature: float = 0.0) -> Dict[str, Any]:
    """Query the LLM using retrieved chunks with error handling"""
    if faiss_index is None:
        return {"result": "Error: Document index not available. Please try uploading your document again."}
    
    try:
        # Use environment variable for API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {"result": "Error: Anthropic API key not set"}
        
        # Initialize LLM
        llm = ChatAnthropic(
            model=LLM_MODEL, 
            temperature=temperature,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        
        # Set up retriever with more controls
        retriever = faiss_index.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance - balances relevance with diversity
            search_kwargs={
                "k": 5,         
                "fetch_k": 20,  
                "lambda_mult": 0.7  
            }
        )
        
        # Enhanced prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert assistant trained to answer questions based on specific document content. "
                "Use ONLY the following context to answer the user's question as accurately and completely as possible.\n\n"
                "Context:\n{context}\n\n"
                "If the answer is not contained within the context, respond with: "
                "'I don't see information about that in the document.'\n\n"
                "If the context is insufficient but suggests a partial answer, explain what information is available "
                "and what is missing.\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        # Execute query
        result = qa_chain({"query": query})
        return result
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return {"result": f"Error generating response: {str(e)}"}

def validate_file(uploaded_file) -> bool:
    """Validate uploaded file size and format"""
    if uploaded_file is None:
        return False
    
    # Check file size (in MB)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_PDF_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.1f} MB). Maximum allowed size is {MAX_PDF_SIZE_MB} MB.")
        return False
    
    # Validate file type
    allowed_types = ["application/pdf", "image/png", "image/jpeg", "image/jpg"]
    if uploaded_file.type not in allowed_types:
        st.error(f"Unsupported file type: {uploaded_file.type}. Please upload a PDF or image file.")
        return False
    
    return True

def display_file_info(uploaded_file):
    """Display information about the uploaded file"""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    st.info(f"""
    **File Information:**
    - Name: {uploaded_file.name}
    - Type: {uploaded_file.type}
    - Size: {file_size_mb:.2f} MB
    """)

def display_system_info():
    """Display information about system configuration"""
    with st.expander("System Information"):
        st.markdown("""
        **System Configuration:**
        - Chunk size: {} characters
        - Chunk overlap: {} characters
        - Embedding model: {}
        - LLM model: {}
        - Max PDF size: {} MB
        """.format(CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, LLM_MODEL, MAX_PDF_SIZE_MB))

def reset_session():
    """Reset the session state"""
    for key in list(st.session_state.keys()):
        if key not in ['ANTHROPIC_API_KEY']:
            del st.session_state[key]
    st.experimental_rerun()

def main():
    st.set_page_config(
        page_title="Advanced Document Q&A with Claude",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Advanced Document Q&A with Claude")
    st.markdown("""
    Upload a PDF or image document and ask questions about its content.
    The app will extract text, create an index, and use Claude to provide accurate answers.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter your Anthropic API key:",
            type="password",
            value=st.session_state.get('ANTHROPIC_API_KEY', ''),
            help="Your API key will be stored in the session only"
        )
        st.session_state['ANTHROPIC_API_KEY'] = api_key
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
        # Temperature slider
        temperature = st.slider(
            "Response Temperature:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.1,
            help="Higher values make output more creative but less accurate"
        )
        
        # Reset button
        if st.button("Reset Session", type="primary"):
            reset_session()
        
        # Display system info
        display_system_info()
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to continue.")
        st.stop()
    
    # File upload section
    st.header("1. Upload a Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file or image (jpg, png)", 
        type=["pdf", "png", "jpg", "jpeg"],
        help="Maximum file size: {} MB".format(MAX_PDF_SIZE_MB)
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        if not validate_file(uploaded_file):
            st.stop()
        
        display_file_info(uploaded_file)
        
        # Check if file was already processed in this session
        file_key = uploaded_file.name + str(uploaded_file.size)
        if 'current_file_key' not in st.session_state or st.session_state.get('current_file_key') != file_key:
            st.session_state['current_file_key'] = file_key
            st.session_state.pop('extracted_text', None)
            st.session_state.pop('chunks', None)
            st.session_state.pop('faiss_index', None)
        
        # Extract text if not already done
        if 'extracted_text' not in st.session_state:
            with st.spinner("Extracting text from document..."):
                progress_bar = st.progress(0, text="Starting text extraction...")
                
                if uploaded_file.type == "application/pdf":
                    extracted_text = extract_text_from_pdf(uploaded_file, progress_bar)
                else:  # Image file
                    extracted_text = extract_text_from_image(uploaded_file)
                    progress_bar.progress(1.0, text="Text extraction complete!")
                
                st.session_state['extracted_text'] = extracted_text
                
                # Show text statistics
                word_count = len(extracted_text.split())
                char_count = len(extracted_text)
                st.success(f"✅ Text extraction complete! Extracted {word_count} words ({char_count} characters)")
        
        # Show extracted text in expander
        with st.expander("📝 View Extracted Text"):
            st.text_area(
                "Extracted Text", 
                st.session_state['extracted_text'], 
                height=300
            )
        
        # Chunk text and create index if not already done
        if 'chunks' not in st.session_state or 'faiss_index' not in st.session_state:
            with st.spinner("Preparing document for questions..."):
                # Chunk the text
                chunks = chunk_text(st.session_state['extracted_text'])
                st.session_state['chunks'] = chunks
                
                if chunks:
                    # Create FAISS index
                    faiss_index = create_faiss_index(chunks)
                    st.session_state['faiss_index'] = faiss_index
                    
                    st.success(f"✅ Document ready! Created {len(chunks)} searchable chunks.")
                else:
                    st.error("No text could be extracted from the document.")
                    st.stop()
        
        # Query interface
        st.header("2. Ask Questions About Your Document")
        
        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            Try questions like:
            - "What is the main topic of this document?"
            - "Summarize the key points in this document."
            - "What are the conclusions presented in this document?"
            - "Are there any tables or figures in this document? What do they show?"
            """)
        
        query = st.text_input(
            "Enter your question about the document:",
            key="question_input",
            placeholder="e.g., What is the main topic of this document?"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.button("Ask Claude", type="primary")
        with col2:
            clear_button = st.button("Clear Results")
            
        if clear_button:
            st.session_state.pop('last_query', None)
            st.session_state.pop('last_result', None)
            st.experimental_rerun()
        
        if (submit_button or query) and query and 'faiss_index' in st.session_state:
            # Check if this is a new query
            if 'last_query' not in st.session_state or st.session_state['last_query'] != query:
                st.session_state['last_query'] = query
                
                with st.spinner("🧠 Claude is analyzing your document..."):
                    result = query_llm(
                        st.session_state['faiss_index'], 
                        query,
                        temperature=temperature
                    )
                    st.session_state['last_result'] = result
            else:
                # Use cached result
                result = st.session_state['last_result']
            
            # Display results
            st.subheader("📘 Answer:")
            st.markdown(result["result"])
            
            # Show source chunks in expander
            if "source_documents" in result:
                with st.expander("🔍 Source passages used for this answer"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Passage {i+1}:**")
                        st.markdown(f"```\n{doc.page_content}\n```")
                        st.markdown("---")
    else:
        # Show placeholder when no file is uploaded
        st.info("👆 Please upload a document to get started")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Unhandled exception in application")