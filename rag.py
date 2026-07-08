import json
import faiss
import numpy as np
from PIL import Image
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from model import load_llava_model_4bit
import re
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRAGPipeline:
    def __init__(self):
        self.processor = None
        self.model = None
        self.embedder = None
        self.metadata = None
        self.index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.rag_ready = False

    def initialize_models(self):
        """Initialize LLaVA and embedding models"""
        if self.processor is None:
            logger.info("🔄 Loading LLaVA model...")
            self.processor, self.model = load_llava_model_4bit()
            logger.info("✅ LLaVA model loaded")
        
        if self.embedder is None:
            logger.info("🔄 Loading embedding model...")
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("✅ Embedding model loaded")

    def load_vector_store(self):
        """Load existing vector store and metadata"""
        try:
            # Check for both metadata.json and metadata1.json
            metadata_file = None
            if Path("metadata.json").exists():
                metadata_file = "metadata.json"
            elif Path("metadata1.json").exists():
                metadata_file = "metadata1.json"
            
            if metadata_file and Path("vector_store.faiss").exists():
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
                self.index = faiss.read_index("vector_store.faiss")
                
                logger.info(f"✅ Loaded {len(self.metadata)} chunks from {metadata_file}")
                
                # Setup TF-IDF
                self.setup_tfidf_search()
                self.rag_ready = True
                return True
            else:
                logger.warning("⚠️ No vector store found")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            return False

    def process_uploaded_pdfs(self, pdf_files, progress=gr.Progress()):
        """Process uploaded PDFs through the entire pipeline"""
        try:
            # Create temp directory for PDFs
            temp_pdf_dir = Path("temp_pdfs")
            temp_pdf_dir.mkdir(exist_ok=True)
            
            # Clear existing files
            for file in temp_pdf_dir.glob("*.pdf"):
                file.unlink()
            
            # Save uploaded PDFs
            if pdf_files is None or len(pdf_files) == 0:
                return "❌ No PDF files uploaded!"
            
            progress(0, desc="📁 Saving uploaded PDFs...")
            for pdf_file in pdf_files:
                pdf_path = Path(pdf_file.name)
                dest_path = temp_pdf_dir / pdf_path.name
                shutil.copy(pdf_file.name, dest_path)
                logger.info(f"✅ Saved: {pdf_path.name}")
            
            total_steps = 3
            
            # Step 1: Run markdown.py
            progress(0.25, desc="📝 Step 1/3: Converting PDFs to Markdown...")
            logger.info("🔄 Running markdown.py...")
            result = subprocess.run(
                [sys.executable, "markdown.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            # Log output for debugging
            if result.stdout:
                logger.info(f"Markdown.py output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Markdown.py stderr: {result.stderr}")
            
            if result.returncode != 0:
                error_msg = f"❌ Markdown conversion failed with return code {result.returncode}"
                logger.error(error_msg)
                return error_msg
            
            # Check if markdown files were created
            output_dir = Path("outputs")
            if not output_dir.exists() or not list(output_dir.glob("*/*.md")):
                error_msg = "❌ Markdown conversion failed: No markdown files created"
                logger.error(error_msg)
                return error_msg
            
            logger.info("✅ Markdown conversion completed")
            
            # Step 2: Run chunks.py
            progress(0.5, desc="🔪 Step 2/3: Creating chunks...")
            logger.info("🔄 Running chunks.py...")
            result = subprocess.run(
                [sys.executable, "chunks.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            # Log output for debugging
            if result.stdout:
                logger.info(f"Chunks.py output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Chunks.py stderr: {result.stderr}")
            
            if result.returncode != 0:
                error_msg = f"❌ Chunking failed with return code {result.returncode}"
                logger.error(error_msg)
                return error_msg
            
            # Check if chunks file was created
            chunks_file = Path("all_chunks.json")
            if not chunks_file.exists():
                error_msg = "❌ Chunking failed: all_chunks.json not created"
                logger.error(error_msg)
                return error_msg
            
            logger.info("✅ Chunking completed")
            
            # Step 3: Run embedding.py
            progress(0.75, desc="🧠 Step 3/3: Creating embeddings...")
            logger.info("🔄 Running embedding.py...")
            result = subprocess.run(
                [sys.executable, "embedding.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            # Log output for debugging
            if result.stdout:
                logger.info(f"Embedding.py output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Embedding.py stderr: {result.stderr}")
            
            if result.returncode != 0:
                error_msg = f"❌ Embedding creation failed with return code {result.returncode}"
                logger.error(error_msg)
                return error_msg
            
            # Check if embedding files were created (check both metadata.json and metadata1.json)
            metadata_exists = Path("metadata.json").exists() or Path("metadata1.json").exists()
            if not Path("vector_store.faiss").exists() or not metadata_exists:
                error_msg = "❌ Embedding creation failed: vector_store.faiss or metadata file not created"
                logger.error(error_msg)
                return error_msg
            
            logger.info("✅ Embedding creation completed")
            
            # Step 4: Load the new vector store
            progress(0.9, desc="📚 Loading vector store...")
            self.initialize_models()
            if not self.load_vector_store():
                return "❌ Failed to load vector store after processing"
            
            progress(1.0, desc="✅ Processing complete!")
            
            num_pdfs = len(pdf_files)
            num_chunks = len(self.metadata) if self.metadata else 0
            
            success_msg = f"""
✅ **Processing Complete!**

📊 **Statistics:**
- PDFs Processed: {num_pdfs}
- Chunks Created: {num_chunks}
- Vector Store: Ready

🎯 **You can now ask questions!**
"""
            return success_msg
            
        except Exception as e:
            error_msg = f"❌ Error during processing: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def setup_tfidf_search(self):
        """Setup TF-IDF search for better keyword matching"""
        logger.info("Setting up TF-IDF search...")
        
        all_texts = []
        for chunk in self.metadata:
            combined_text = self.prepare_text_for_search(chunk)
            all_texts.append(combined_text)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            sublinear_tf=True,
            norm='l2'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def prepare_text_for_search(self, chunk: Dict) -> str:
        """Consistent text preparation for both embedding and TF-IDF"""
        text = chunk.get("text", "").strip()
        pdf_name = chunk.get("pdf", "").strip()
        
        combined_text = f"{pdf_name}. {text}" if pdf_name else text
        return combined_text

    def resize_image(self, image: Image.Image):
        patch_size = self.model.config.vision_config.patch_size
        shortest_edge = self.processor.image_processor.size.get("shortest_edge", 336)

        orig_w, orig_h = image.size
        scale = shortest_edge / min(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        new_w = (new_w // patch_size) * patch_size
        new_h = (new_h // patch_size) * patch_size

        return image.resize((new_w, new_h))

    def preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        query = ' '.join(query.split())
        query = re.sub(r'^(what|how|where|when|why|which|who)\s+(is|are|was|were|does|do|did|can|could|should|would)\s*', '', query, flags=re.IGNORECASE)
        return query.strip()

    def extract_important_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        stop_words = {
            'the', 'of', 'and', 'are', 'is', 'in', 'to', 'for', 'a', 'an', 'with', 'by', 'from', 
            'on', 'at', 'as', 'what', 'how', 'where', 'when', 'why', 'which', 'that', 'this'
        }
        
        query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
        words = query_clean.split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords

    def calculate_keyword_score(self, keywords: List[str], chunk: Dict) -> float:
        """Calculate keyword matching score"""
        if not keywords:
            return 0.0
        
        text = chunk.get("text", "").lower()
        
        if not text.strip():
            return 0.0
        
        exact_matches = 0
        partial_matches = 0
        
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                exact_matches += 1
            elif keyword in text:
                partial_matches += 1
        
        total_weight = len(keywords)
        score = (exact_matches * 1.0 + partial_matches * 0.5) / total_weight
        
        return min(score, 1.0)

    def is_valid_chunk(self, chunk: Dict) -> bool:
        """Check if chunk contains meaningful content"""
        text = chunk.get("text", "").strip()
        
        if len(text) < 30:
            return False
        
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count < 8:
            return False
        
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if special_char_ratio > 0.3:
            return False
        
        return True

    def tfidf_first_search(self, query: str, top_k: int = 20) -> Dict:
        """Search using TF-IDF priority"""
        logger.info(f"🔍 Searching for: '{query}'")
        
        query_processed = self.preprocess_query(query)
        
        # TF-IDF scores
        query_tfidf = self.tfidf_vectorizer.transform([query_processed])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # FAISS candidates
        query_vector = self.embedder.encode([query_processed])
        query_vector = np.array(query_vector).astype("float32")
        
        search_k = min(top_k * 4, len(self.metadata))
        faiss_distances, faiss_indices = self.index.search(query_vector, search_k)
        faiss_candidates = set(faiss_indices[0])
        
        # Keywords
        query_keywords = self.extract_important_keywords(query)
        
        candidates = []
        
        for idx in range(len(self.metadata)):
            chunk = self.metadata[idx]
            
            if not self.is_valid_chunk(chunk):
                continue
                
            tfidf_score = tfidf_scores[idx] if idx < len(tfidf_scores) else 0.0
            
            if tfidf_score > 0.01:
                faiss_bonus = 0.1 if idx in faiss_candidates else 0.0
                keyword_score = self.calculate_keyword_score(query_keywords, chunk)
                keyword_bonus = keyword_score * 0.05
                
                final_score = tfidf_score + faiss_bonus + keyword_bonus
                
                candidates.append({
                    'chunk': chunk,
                    'final_score': final_score,
                    'idx': idx
                })
        
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        if candidates:
            return candidates[0]['chunk']
        
        return None

    def create_context(self, chunk: Dict) -> str:
        if not chunk:
            return "No relevant content found."

        text = chunk.get("text", "").strip()
        pdf_name = chunk.get("pdf", "Unknown Document")

        context = f"Document: {pdf_name}\n\nContent:\n{text}\n"
        return context.strip()

    def clean_answer(self, answer: str) -> str:
        answer = re.sub(r'\[\d+\]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer

    def generate_answer(self, query: str, chunk: Dict) -> str:
        if not chunk:
            return "No relevant document section found for your query."

        context = self.create_context(chunk)
        image_paths = chunk.get("images", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        valid_images = []
        for img_path in image_paths:
            try:
                img = self.resize_image(Image.open(img_path).convert("RGB"))
                valid_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")

        if valid_images:
            answer = self.generate_with_images(query, context, valid_images)
            if "not provide enough information" in answer or len(answer) < 50:
                answer = self.generate_text_only(query, context)
            return self.clean_answer(answer)

        answer = self.generate_text_only(query, context)
        return self.clean_answer(answer)

    def generate_text_only(self, query: str, context: str) -> str:
        prompt = f"""Answer the question using ONLY the information provided in the context below.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Use only information from the context above
- Your answer must be explainable using information from the context
- If the context doesn't contain the answer, say so clearly
- Be specific and quote relevant parts when possible
- Do NOT include citation markers like [15], [26], etc.

ANSWER:"""

        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        input_length = inputs["input_ids"].shape[1]
        response_tokens = output[0][input_length:]
        response = self.processor.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return self.clean_answer(response.strip())

    def generate_with_images(self, query: str, context: str, images: List[Image.Image]) -> str:
        image_tokens = "<image>" * len(images)
        prompt = f"""Answer the question using the provided images and text context.

{image_tokens}

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Analyze the images carefully and describe what you see
- Use only information from the context and images
- Do NOT include citation markers like [15], [26], etc.

ANSWER:"""

        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        input_length = inputs["input_ids"].shape[1]
        response_tokens = output[0][input_length:]
        response = self.processor.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return self.clean_answer(response.strip())

    def pipeline(self, user_query: str) -> Tuple[str, str, str, List[Optional[Image.Image]]]:
        """Main pipeline for answering questions"""
        if not self.rag_ready:
            return "⚠️ Please upload and process PDFs first!", "No PDF", "No content", []

        if not user_query.strip():
            return "Please enter a question.", "No PDF selected", "No text content", []

        try:
            best_chunk = self.tfidf_first_search(user_query, top_k=20)

            if not best_chunk:
                return "No relevant information found in the documents.", "No PDF found", "No content", []

            pdf_name = best_chunk.get("pdf", "Unknown Document")
            text_content = best_chunk.get("text", "No text content available")
            image_paths = best_chunk.get("images", [])
            
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            loaded_images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    loaded_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image: {e}")
            
            answer = self.generate_answer(user_query, best_chunk)
            
            return answer, pdf_name, text_content, loaded_images

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return f"Error: {str(e)}", "Error", "Error", []


# ========== Gradio UI ==========
if __name__ == "__main__":
    rag_pipeline = SimpleRAGPipeline()

    def process_pdfs(pdf_files, progress=gr.Progress()):
        """Handle PDF upload and processing"""
        return rag_pipeline.process_uploaded_pdfs(pdf_files, progress)

    def ask_question(user_query):
        """Handle question answering"""
        answer, pdf_name, text_content, loaded_images = rag_pipeline.pipeline(user_query)
        return answer, pdf_name, text_content, loaded_images

    # Create Gradio interface
    with gr.Blocks(title="RAG Pipeline with Auto-Processing") as interface:
        
        gr.Markdown("# 📚 RAG Pipeline with Automatic Processing")
        gr.Markdown("""
        **How to use:**
        1. Upload your PDF files (multiple files supported)
        2. Click "Process PDFs" and wait for completion
        3. Once processing is done, ask your questions!
        """)
        
        # PDF Upload Section
        gr.Markdown("## 📁 Step 1: Upload PDFs")
        with gr.Row():
            pdf_upload = gr.File(
                label="Upload PDF Files",
                file_count="multiple",
                file_types=[".pdf"],
                scale=3
            )
        
        with gr.Row():
            process_btn = gr.Button("🚀 Process PDFs", variant="primary", size="lg")
        
        with gr.Row():
            processing_status = gr.Textbox(
                label="Processing Status",
                lines=8,
                interactive=False
            )
        
        gr.Markdown("---")
        
        # Question Answering Section
        gr.Markdown("## 💬 Step 2: Ask Questions")
        
        with gr.Row():
            user_input = gr.Textbox(
                lines=3,
                placeholder="Type your question here...",
                label="Your Question",
                scale=4
            )
        
        with gr.Row():
            submit_btn = gr.Button("Ask Question", variant="primary", scale=1)
        
        gr.Markdown("## 🤖 AI Answer")
        with gr.Row():
            answer_output = gr.Textbox(label="Response", lines=8, scale=1)
        
        gr.Markdown("## 📄 Source Information")
        with gr.Row():
            pdf_output = gr.Textbox(label="📋 PDF Document", lines=1, scale=1)
            
        with gr.Row():
            text_output = gr.Textbox(label="📝 Text Content", lines=6, scale=1)
        
        gr.Markdown("## 🖼️ Related Images")
        with gr.Row():
            images_gallery = gr.Gallery(
                label="Images from Selected Chunk",
                show_label=True,
                columns=3,
                rows=2,
                object_fit="contain",
                height="auto"
            )
        
        # Connect functions
        process_btn.click(
            fn=process_pdfs,
            inputs=[pdf_upload],
            outputs=[processing_status]
        )
        
        submit_btn.click(
            fn=ask_question,
            inputs=[user_input],
            outputs=[answer_output, pdf_output, text_output, images_gallery]
        )
        
        user_input.submit(
            fn=ask_question,
            inputs=[user_input],
            outputs=[answer_output, pdf_output, text_output, images_gallery]
        )
    
    interface.launch(share=True)