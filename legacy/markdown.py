import logging
import time
import re
import hashlib
import gc
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)
IMAGE_RESOLUTION_SCALE = 2.0  # Further reduced for memory efficiency
BATCH_SIZE = 10  # Process 10 files then restart
RESTART_AFTER_FILES = 30  # Restart process after 30 files
MAX_MEMORY_GB = 8  # Restart if memory usage exceeds this


def sanitize_filename(name: str) -> str:
    """Sanitize file/folder names by replacing spaces and special characters."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())


def shorten_filename(name: str, max_prefix: int = 10, hash_length: int = 8) -> str:
    """
    Keep first few characters and append hash to keep uniqueness.
    """
    name = sanitize_filename(name)
    prefix = name[:max_prefix]
    name_hash = hashlib.md5(name.encode("utf-8")).hexdigest()[:hash_length]
    return f"{prefix}_{name_hash}"


def get_memory_usage():
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
    except:
        return 0


def clear_gpu_memory():
    """Clear GPU memory if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass


def load_checkpoint(checkpoint_file: Path) -> List[str]:
    """Load processed files from checkpoint."""
    if checkpoint_file.exists():
        try:
            with checkpoint_file.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except:
            return []
    return []


def save_checkpoint(checkpoint_file: Path, processed_files: List[str]):
    """Save processed files to checkpoint."""
    try:
        with checkpoint_file.open("w", encoding="utf-8") as f:
            for filename in processed_files:
                f.write(f"{filename}\n")
    except Exception as e:
        _log.error(f"Failed to save checkpoint: {e}")


def rename_referenced_images_and_update_md(md_file: Path, image_dir: Path):
    """Rename long image filenames and update markdown references."""
    if not md_file.exists() or not image_dir.exists():
        return

    try:
        with md_file.open("r", encoding="utf-8") as f:
            content = f.read()

        image_pattern = re.compile(r'!\[Image\]\(([^)]+)\)')
        matches = image_pattern.findall(content)

        new_content = content
        for idx, old_path in enumerate(matches, 1):
            old_filename = Path(old_path).name
            new_filename = f"img_{idx:03d}.png"

            old_image_path = image_dir / old_filename
            new_image_path = image_dir / new_filename
            if old_image_path.exists():
                old_image_path.rename(new_image_path)

            new_content = new_content.replace(old_path, f"{image_dir.name}/{new_filename}")

        with md_file.open("w", encoding="utf-8") as f:
            f.write(new_content)

    except Exception as e:
        _log.error(f"Error processing markdown file {md_file}: {e}")


def process_single_pdf(pdf_path: Path, output_dir: Path) -> bool:
    """Process a single PDF file with memory management."""
    doc_converter = None
    conv_res = None
    
    try:
        _log.info(f"🔄 Processing: {pdf_path.name}")
        
        # Check memory before processing
        memory_usage = get_memory_usage()
        if memory_usage > MAX_MEMORY_GB:
            _log.warning(f"⚠️ High memory usage: {memory_usage:.2f}GB")
            clear_gpu_memory()
            gc.collect()
        
        original_filename = pdf_path.stem
        safe_filename = shorten_filename(original_filename)
        safe_output_dir = output_dir / safe_filename
        safe_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lightweight pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        start_time = time.time()
        conv_res = doc_converter.convert(pdf_path)

        # Extract images with error handling
        table_counter = 0
        picture_counter = 0
        
        for element, _ in conv_res.document.iterate_items():
            try:
                if isinstance(element, TableItem):
                    table_counter += 1
                    image_file = safe_output_dir / f"{safe_filename}-table-{table_counter}.png"
                    image_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Check if image exists before saving
                    img = element.get_image(conv_res.document)
                    if img is not None:
                        with image_file.open("wb") as fp:
                            img.save(fp, "PNG")
                        _log.info(f"💾 Saved table {table_counter}")
                    else:
                        _log.warning(f"⚠️ Table {table_counter} image is None")
                        
                elif isinstance(element, PictureItem):
                    picture_counter += 1
                    image_file = safe_output_dir / f"{safe_filename}-picture-{picture_counter}.png"
                    image_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Check if image exists before saving
                    img = element.get_image(conv_res.document)
                    if img is not None:
                        with image_file.open("wb") as fp:
                            img.save(fp, "PNG")
                        _log.info(f"💾 Saved picture {picture_counter}")
                    else:
                        _log.warning(f"⚠️ Picture {picture_counter} image is None")
                        
            except Exception as img_error:
                _log.warning(f"⚠️ Failed to extract image: {img_error}")
                continue

        # Save markdown files
        md_embedded = safe_output_dir / f"{safe_filename}-embedded.md"
        md_referenced = safe_output_dir / f"{safe_filename}-referenced.md"

        try:
            conv_res.document.save_as_markdown(md_embedded, image_mode=ImageRefMode.EMBEDDED)
        except Exception as e:
            _log.warning(f"⚠️ Failed to save embedded markdown: {e}")

        try:
            conv_res.document.save_as_markdown(md_referenced, image_mode=ImageRefMode.REFERENCED)
            referenced_artifacts_dir = safe_output_dir / f"{safe_filename}-referenced_artifacts"
            if referenced_artifacts_dir.exists():
                rename_referenced_images_and_update_md(md_referenced, referenced_artifacts_dir)
        except Exception as e:
            _log.warning(f"⚠️ Failed to save referenced markdown: {e}")

        elapsed = time.time() - start_time
        _log.info(f"✅ Completed '{pdf_path.name}' in {elapsed:.2f}s")
        
        return True

    except Exception as e:
        _log.error(f"❌ Error processing {pdf_path.name}: {e}")
        return False
    
    finally:
        # Aggressive cleanup
        if conv_res:
            del conv_res
        if doc_converter:
            del doc_converter
        clear_gpu_memory()
        gc.collect()


def restart_process(script_path: str, start_index: int):
    """Restart the process from a specific file index."""
    _log.info(f"🔄 Restarting process from index {start_index}")
    
    # Clear all caches before restart
    clear_gpu_memory()
    gc.collect()
    
    # Restart with start index
    cmd = [sys.executable, script_path, "--start-index", str(start_index)]
    subprocess.run(cmd)
    sys.exit(0)


def process_files_with_restart(pdf_files: List[Path], output_dir: Path, start_index: int = 0):
    """Process files with automatic restart capability."""
    checkpoint_file = output_dir / "checkpoint.txt"
    processed_files = load_checkpoint(checkpoint_file)
    
    total_files = len(pdf_files)
    processed_count = len(processed_files)
    
    _log.info(f"📊 Total files: {total_files}, Already processed: {processed_count}")
    
    for i in range(start_index, total_files):
        pdf_file = pdf_files[i]
        
        # Skip if already processed
        if pdf_file.name in processed_files:
            _log.info(f"⏭️ Skipping already processed: {pdf_file.name}")
            continue
        
        # Process the file
        success = process_single_pdf(pdf_file, output_dir)
        
        if success:
            processed_files.append(pdf_file.name)
            processed_count += 1
            
            # Save checkpoint after each successful file
            save_checkpoint(checkpoint_file, processed_files)
            
            _log.info(f"📈 Progress: {processed_count}/{total_files} ({(processed_count/total_files)*100:.1f}%)")
        
        # Check if we need to restart (more frequent restarts)
        if (i + 1) % RESTART_AFTER_FILES == 0 and i + 1 < total_files:
            _log.info(f"🔄 Auto-restarting after {RESTART_AFTER_FILES} files for memory cleanup...")
            time.sleep(3)  # Wait a bit before restart
            restart_process(__file__, i + 1)
        
        # Memory check (more aggressive)
        memory_usage = get_memory_usage()
        if memory_usage > MAX_MEMORY_GB:
            _log.warning(f"⚠️ Memory usage {memory_usage:.2f}GB > {MAX_MEMORY_GB}GB - Restarting...")
            time.sleep(3)
            restart_process(__file__, i + 1)
            
        # Small delay between files
        time.sleep(1)
    
    _log.info(f"🎉 All files processed! Total successful: {processed_count}/{total_files}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PDF files with auto-restart')
    parser.add_argument('--start-index', type=int, default=0, help='Start processing from this file index')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdf_processing.log'),
            logging.StreamHandler()
        ]
    )

    input_dir = Path("temp_pdfs")
    output_base_dir = Path("outputs")
    output_base_dir.mkdir(exist_ok=True)

    # Get all PDF files
    pdf_files = sorted(list(input_dir.glob("*.pdf")))
    
    if not pdf_files:
        _log.error("❌ No PDF files found in the input directory!")
        return

    _log.info(f"📁 Found {len(pdf_files)} PDF files")
    
    # Process files with restart capability
    process_files_with_restart(pdf_files, output_base_dir, args.start_index)


if __name__ == "__main__":
    main()