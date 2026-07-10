"""Central configuration for the RAG pipeline. All models are local/open-source."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Models (HuggingFace Hub IDs — downloaded once, cached locally)
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GENERATOR_MODEL = "Qwen/Qwen2.5-14B-Instruct"
GENERATOR_PROVIDER = "huggingface"   # any LangChain provider: "huggingface", "ollama", ...
GENERATOR_LOAD_IN_4BIT = True        # ~8GB instead of ~28GB; leaves room for the vision model
                                     # (bge-m3 + reranker also hold ~5GB GPU) on a 40GB A100
MAX_NEW_TOKENS = 512

# Vision (figure-grounded answers)
VISION_ENABLED = True                # route figure questions to the vision model (needs GPU)
VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VISION_LOAD_IN_4BIT = True           # ~6GB — fits next to the resident 14B generator on a 40GB A100
MAX_FIGURES_PER_ANSWER = 4           # most figures passed to the vision model in one prompt

# Ingestion
OCR_ENABLED = False             # born-digital PDFs need no OCR; enable for scanned documents
CHUNK_MAX_TOKENS = 512          # HybridChunker target; bge-m3 handles up to 8192
PDF_DIR = REPO_ROOT / "data" / "pdfs"
DOCLING_CACHE = REPO_ROOT / "data" / "docling"   # converted DoclingDocument JSON, keyed by content hash
FIGURES_DIR = REPO_ROOT / "data" / "figures"     # extracted figure images, {doc_hash}-fig{n}.png
EXCLUDED_HEADINGS = {"references", "table of contents"}  # keep abstract and acknowledgments — both answer real queries

# Vector store
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "production_rag"

# Cross-document questions: fan retrieval out over every document in scope
MULTI_DOC_FANOUT = True
PER_DOC_TOP_K = 2               # chunks each document contributes, after the main retrieval
MULTI_DOC_MAX_DOCS = 8          # above this, fan out only over documents the main retrieval surfaced

# Retrieval
DENSE_TOP_K = 25                # candidates from the dense index
BM25_TOP_K = 25                 # candidates from BM25
RERANK_TOP_N = 5                # chunks passed to the generator after reranking
RERANK_BLEND_LAMBDA = 2.0       # final = sigmoid(rerank) + lambda * RRF; measured sweep in eval/results
