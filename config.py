"""Central configuration for the RAG pipeline. All models are local/open-source."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Models (HuggingFace Hub IDs — downloaded once, cached locally)
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GENERATOR_MODEL = "Qwen/Qwen2.5-14B-Instruct"
GENERATOR_PROVIDER = "huggingface"   # any LangChain provider: "huggingface", "ollama", ...
MAX_NEW_TOKENS = 512

# Ingestion
OCR_ENABLED = False             # born-digital PDFs need no OCR; enable for scanned documents
CHUNK_MAX_TOKENS = 512          # HybridChunker target; bge-m3 handles up to 8192
PDF_DIR = REPO_ROOT / "data" / "pdfs"
DOCLING_CACHE = REPO_ROOT / "data" / "docling"   # converted DoclingDocument JSON, keyed by content hash
EXCLUDED_HEADINGS = {"references", "table of contents"}  # keep abstract and acknowledgments — both answer real queries

# Vector store
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "production_rag"

# Retrieval
DENSE_TOP_K = 25                # candidates from the dense index
BM25_TOP_K = 25                 # candidates from BM25
RERANK_TOP_N = 5                # chunks passed to the generator after reranking
RERANK_BLEND_LAMBDA = 2.0       # final = sigmoid(rerank) + lambda * RRF; measured sweep in eval/results
