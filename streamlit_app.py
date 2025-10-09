import streamlit as st
import os
from pathlib import Path
from typing import List
import spacy
from semantic_search_mcp.chunkers import make_chunker

# Configure page
st.set_page_config(
    page_title="Sentence Splitter Demo",
    layout="wide"
)

class SpacySentenceSplitter:
    """Custom SpaCy sentence splitter to match LangChain interface"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
        except IOError:
            st.error(f"SpaCy model '{model_name}' not found. Please install it using: python -m spacy download {model_name}")
            self.nlp = None
    
    def split_text(self, text: str) -> List[str]:
        """Split text into sentences using SpaCy"""
        if self.nlp is None:
            return [text]
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def get_name(self) -> str:
        return f"SpaCy Sentence Splitter ({self.model_name})"

def load_text_file(file_path: str) -> str:
    """Load text content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return ""

def get_corpus_files() -> List[str]:
    """Get list of text files from the corpus folder"""
    corpus_path = Path("data/corpus")
    if not corpus_path.exists():
        return []
    
    text_files = []
    for file_path in corpus_path.glob("*.txt"):
        text_files.append(str(file_path))
    
    return sorted(text_files)

def main():
    st.title("📝 Sentence Splitter Demo")
    st.markdown("Compare different sentence splitting approaches using SpaCy and LangChain text splitters.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    splitter_options = {
        "SpaCy Sentence Splitter": "spacy",
        "Recursive Character Text Splitter": "recursive",
        "Custom Splitter": "custom",
    }
    
    selected_splitter = st.sidebar.selectbox(
        "Select Sentence Splitter:",
        options=list(splitter_options.keys())
    )
    
    # Get corpus files
    corpus_files = get_corpus_files()
    
    if not corpus_files:
        st.error("No text files found in data/corpus/ directory")
        return
    
    # File selection
    selected_file = st.sidebar.selectbox(
        "Select Text File:",
        options=corpus_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Additional parameters for RecursiveCharacterTextSplitter
    if splitter_options[selected_splitter] == "recursive":
        chunk_size = st.sidebar.slider(
            "Chunk Size:",
            min_value=50,
            max_value=2000,
            value=500,
            step=50
        )
        chunk_overlap = st.sidebar.slider(
            "Chunk Overlap:",
            min_value=0,
            max_value=200,
            value=50,
            step=10
        )
    
    # Apply button
    apply_clicked = st.sidebar.button("Apply", type="primary")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Selected Configuration")
        st.write(f"**Splitter:** {selected_splitter}")
        st.write(f"**File:** {os.path.basename(selected_file)}")
        
        if splitter_options[selected_splitter] == "recursive":
            st.write(f"**Chunk Size:** {chunk_size}")
            st.write(f"**Chunk Overlap:** {chunk_overlap}")
    
    with col2:
        if apply_clicked:
            # Load the selected text file
            text_content = load_text_file(selected_file)
            
            if not text_content:
                st.error("Could not load the selected file or file is empty.")
                return
            
            # Display original text preview
            with st.expander("Original Text Preview"):
                st.text_area(
                    "Content:",
                    value=text_content[:1000] + ("..." if len(text_content) > 1000 else ""),
                    height=150,
                    disabled=True
                )
            
            # Initialize the selected splitter
            try:
                chunker = make_chunker(spec={
                    "type": splitter_options[selected_splitter],
                    "spacy_model": "en_core_web_sm",
                    "chunk_size": 1000,
                    "chunk_overlap": 100,
                })
                sentences = chunker.split_text(text_content)
                # if splitter_options[selected_splitter] == "spacy":
                #     splitter = SpacySentenceSplitter()
                #     if splitter.nlp is None:
                #         return
                #     sentences = splitter.split_text(text_content)
                    
                # elif splitter_options[selected_splitter] == "recursive":
                #     splitter = RecursiveCharacterTextSplitter(
                #         chunk_size=chunk_size,
                #         chunk_overlap=chunk_overlap,
                #         length_function=len,
                #         separators=["\n\n", "\n", ". ", " ", ""]
                #     )
                #     sentences = splitter.split_text(text_content)
                
                # Display results
                st.subheader("Split Results")
                st.write(f"**Total sentences/chunks:** {len(sentences)}")
                
                # Display sentences in a scrollable container
                st.markdown("### Sentences/Chunks:")
                
                for i, sentence in enumerate(sentences, 1):
                    with st.container():
                        st.markdown(f"**{i}.** {sentence}")
                        st.divider()
                
                # Statistics
                if sentences:
                    avg_length = sum(len(s) for s in sentences) / len(sentences)
                    min_length = min(len(s) for s in sentences)
                    max_length = max(len(s) for s in sentences)
                    
                    st.subheader("Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Count", len(sentences))
                    with col2:
                        st.metric("Avg Length", f"{avg_length:.0f}")
                    with col3:
                        st.metric("Min Length", min_length)
                    with col4:
                        st.metric("Max Length", max_length)
                        
            except Exception as e:
                st.error(f"Error processing text with {selected_splitter}: {str(e)}")
        else:
            st.info("👈 Select your configuration and click 'Apply' to see the results!")

if __name__ == "__main__":
    main()
