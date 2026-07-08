import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_FILE = Path("all_chunks.json")
FAISS_INDEX_FILE = Path("vector_store.faiss")
METADATA_FILE = Path("metadata1.json")
MODEL_NAME = "all-MiniLM-L6-v2"  # You can change this to other models

class EmbeddingProcessor:
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the embedding processor with a sentence transformer model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logger.info(f"Embeddings created with shape: {embeddings.shape}")
        return embeddings
    
    def prepare_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare text from chunk for embedding.
        Combines text, table captions, and image information.
        """
        text_parts = []
        
        # Add main text
        if chunk.get("text", "").strip():
            text_parts.append(chunk["text"].strip())
        
        # Add table information (captions and headers)
        if chunk.get("tables"):
            for table in chunk["tables"]:
                if table.get("caption"):
                    text_parts.append(f"Table: {table['caption']}")
                if table.get("headers"):
                    headers_text = " | ".join(table["headers"])
                    text_parts.append(f"Table headers: {headers_text}")
        
        # Add image information
        if chunk.get("images"):
            for img_path in chunk["images"]:
                # Extract filename from path for better semantic meaning
                img_name = Path(img_path).stem.replace("_", " ").replace("-", " ")
                text_parts.append(f"Image: {img_name}")
        
        # Combine all parts
        combined_text = " ".join(text_parts).strip()
        
        # Fallback if no text content
        if not combined_text:
            combined_text = f"Document chunk from {chunk.get('pdf', 'unknown')}"
        
        return combined_text

def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """Load chunks from JSON file."""
    logger.info(f"Loading chunks from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create FAISS index from embeddings."""
    logger.info("Creating FAISS index...")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index (using Inner Product for normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # Add embeddings to index
    index.add(embeddings)
    
    logger.info(f"FAISS index created with {index.ntotal} vectors")
    return index

def save_vector_store(index: faiss.Index, metadata: List[Dict[str, Any]], 
                     index_file: Path, metadata_file: Path):
    """Save FAISS index and metadata to files."""
    logger.info(f"Saving FAISS index to: {index_file}")
    faiss.write_index(index, str(index_file))
    
    logger.info(f"Saving metadata to: {metadata_file}")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info("Vector store saved successfully!")

def load_vector_store(index_file: Path, metadata_file: Path):
    """Load FAISS index and metadata from files."""
    logger.info(f"Loading FAISS index from: {index_file}")
    index = faiss.read_index(str(index_file))
    
    logger.info(f"Loading metadata from: {metadata_file}")
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    logger.info(f"Vector store loaded: {index.ntotal} vectors, {len(metadata)} metadata entries")
    return index, metadata

def search_similar(query: str, index: faiss.Index, metadata: List[Dict[str, Any]], 
                  model: SentenceTransformer, top_k: int = 5):
    """Search for similar chunks given a query."""
    logger.info(f"Searching for: '{query}' (top {top_k} results)")
    
    # Create query embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:  # Valid result
            result = {
                "rank": i + 1,
                "score": float(score),
                "metadata": metadata[idx]
            }
            results.append(result)
    
    return results

def print_search_results(results: List[Dict]):
    """Print search results in a readable format."""
    for result in results:
        print(f"\n--- Rank {result['rank']} (Score: {result['score']:.4f}) ---")
        metadata = result['metadata']
        print(f"Chunk ID: {metadata['chunk_id']}")
        print(f"PDF: {metadata['pdf']}")
        print(f"Text: {metadata['text'][:200]}{'...' if len(metadata['text']) > 200 else ''}")
        
        if metadata.get('tables'):
            print(f"Tables: {len(metadata['tables'])} table(s)")
            for i, table in enumerate(metadata['tables']):
                print(f"  Table {i+1}: {table.get('caption', 'No caption')}")
        
        if metadata.get('images'):
            print(f"Images: {len(metadata['images'])} image(s)")
            for img in metadata['images']:
                print(f"  - {img}")

def main():
    """Main function to create embeddings and vector store."""
    try:
        # Check if input file exists
        if not INPUT_FILE.exists():
            logger.error(f"Input file not found: {INPUT_FILE}")
            logger.error("Please run the chunking script first to generate the chunks.")
            return
        
        # Load chunks
        chunks = load_chunks(INPUT_FILE)
        
        # Initialize embedding processor
        processor = EmbeddingProcessor()
        
        # Prepare texts for embedding
        logger.info("Preparing texts for embedding...")
        texts_for_embedding = []
        for chunk in chunks:
            text = processor.prepare_text_for_embedding(chunk)
            texts_for_embedding.append(text)
        
        # Create embeddings
        embeddings = processor.create_embeddings(texts_for_embedding)
        
        # Create FAISS index
        index = create_faiss_index(embeddings)
        
        # Save vector store (metadata contains complete chunk information)
        save_vector_store(index, chunks, FAISS_INDEX_FILE, METADATA_FILE)
        
        logger.info("✅ Embedding process completed successfully!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   - Total chunks processed: {len(chunks)}")
        logger.info(f"   - Embedding dimension: {processor.embedding_dim}")
        logger.info(f"   - FAISS index size: {index.ntotal} vectors")
        logger.info(f"   - Files created: {FAISS_INDEX_FILE}, {METADATA_FILE}")
        
        # Demo search functionality
        print("\n" + "="*50)
        print("🔍 DEMO SEARCH FUNCTIONALITY")
        print("="*50)
        
        demo_queries = [
            "machine learning algorithms",
            "data analysis",
            "research methodology"
        ]
        
        for query in demo_queries:
            print(f"\n🔎 Demo search for: '{query}'")
            results = search_similar(query, index, chunks, processor.model, top_k=3)
            if results:
                print_search_results(results)
            else:
                print("No results found.")
            print("-" * 40)
    
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

def search_interface():
    """Interactive search interface."""
    try:
        # Load existing vector store
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            logger.error("Vector store files not found. Please run main() first.")
            return
        
        index, metadata = load_vector_store(FAISS_INDEX_FILE, METADATA_FILE)
        model = SentenceTransformer(MODEL_NAME)
        
        print("\n" + "="*50)
        print("🔍 INTERACTIVE SEARCH INTERFACE")
        print("="*50)
        print("Type your queries (type 'quit' to exit)")
        
        while True:
            query = input("\n🔎 Enter your search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = search_similar(query, index, metadata, model, top_k=5)
            if results:
                print_search_results(results)
            else:
                print("No results found.")
    
    except Exception as e:
        logger.error(f"Error in search interface: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        search_interface()
    else:
        main()