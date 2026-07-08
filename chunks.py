import os
import re
import json
from pathlib import Path

# Constants
EXCLUDED_SECTIONS = {
    "abstract", "references", "table of contents", "list of tables", "list of figures"
}
SECTION_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")

TABLE_PATTERN = re.compile(
    r"(Table[^\n]*)\n(?:\n*)"
    r"((?:\|.*\|\n?)+)",
    re.IGNORECASE
)

IMAGE_PATTERN = re.compile(r"!\[.*?\]\(\s*<?(.*?)>?\s*\)")

# Chunking parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Base directory
BASE_DIR = Path("outputs")
OUTPUT_FILE = Path("all_chunks.json")

def normalize(text):
    return text.strip().lower()

def extract_sections(md_text):
    """Split markdown text into sections by headings."""
    sections = []
    current = {"title": None, "content": []}
    for line in md_text.splitlines():
        match = SECTION_PATTERN.match(line)
        if match:
            if current["title"]:
                sections.append(current)
            current = {"title": match.group(2).strip(), "content": []}
        else:
            current["content"].append(line)
    if current["title"]:
        sections.append(current)
    return sections

def extract_tables(text):
    """Find markdown tables and convert them to structured format (without markdown)."""
    tables = []
    for match in TABLE_PATTERN.finditer(text):
        caption = match.group(1).strip()
        table_markdown = match.group(2).strip()
        lines = [line.strip() for line in table_markdown.splitlines() if line.strip()]

        if len(lines) < 2:
            continue  # Not a valid table

        rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in lines]

        headers = rows[0]
        if re.fullmatch(r"[-:\s|]+", lines[1]):
            data_rows = rows[2:]
        else:
            data_rows = rows[1:]

        structured = {
            "caption": caption,
            "headers": headers,
            "rows": data_rows
        }
        tables.append(structured)
    return tables

def extract_images(text):
    """Find image paths."""
    return IMAGE_PATTERN.findall(text)

def find_table_positions(text):
    """Find start and end positions of tables in text."""
    positions = []
    for match in TABLE_PATTERN.finditer(text):
        positions.append((match.start(), match.end()))
    return positions

def find_image_positions(text):
    """Find start and end positions of images in text."""
    positions = []
    for match in IMAGE_PATTERN.finditer(text):
        positions.append((match.start(), match.end()))
    return positions

def is_position_in_ranges(pos, ranges):
    """Check if position is within any of the given ranges."""
    for start, end in ranges:
        if start <= pos <= end:
            return True
    return False

def recursive_character_split(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into chunks while preserving complete tables and images.
    """
    if len(text) <= chunk_size:
        return [text]
    
    # Find positions of tables and images
    table_positions = find_table_positions(text)
    image_positions = find_image_positions(text)
    protected_ranges = table_positions + image_positions
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Find the best split point
        split_point = end
        
        # Try to find a good split point (sentence boundary, paragraph, etc.)
        # Look backwards from the end for a good split point
        for i in range(end, max(start + chunk_size // 2, start + 1), -1):
            # Check if this position is within a protected range
            if is_position_in_ranges(i, protected_ranges):
                continue
            
            # Prefer splitting at paragraph breaks
            if i > 0 and text[i-1:i+1] == '\n\n':
                split_point = i
                break
            # Then at sentence ends
            elif i > 0 and text[i-1] in '.!?' and text[i] in ' \n':
                split_point = i
                break
            # Then at line breaks
            elif i > 0 and text[i-1] == '\n':
                split_point = i
                break
        
        # If we couldn't find a good split point, check if we're in a protected range
        if split_point == end:
            # Check if the split point is in a protected range
            for range_start, range_end in protected_ranges:
                if range_start <= split_point <= range_end:
                    # Move split point to after the protected range
                    split_point = range_end + 1
                    break
        
        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)
        
        # Calculate next start position with overlap
        next_start = split_point - overlap
        
        # Make sure we don't start in the middle of a protected range
        for range_start, range_end in protected_ranges:
            if range_start <= next_start <= range_end:
                next_start = range_end + 1
                break
        
        start = max(next_start, split_point)
    
    return chunks

def remove_tables(text):
    """Remove tables and captions from text."""
    return TABLE_PATTERN.sub("", text).strip()

def fix_image_paths(md_path: Path, image_paths):
    """
    Converts relative image paths from markdown to paths relative to project root.
    """
    PROJECT_ROOT = Path.cwd()
    fixed_paths = []
    for p in image_paths:
        abs_path = (md_path.parent / p).resolve()
        try:
            rel_path = abs_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = abs_path
        fixed_paths.append(str(rel_path))
    return fixed_paths

def process_md_file(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    chunks = []
    sections = extract_sections(md_text)
    
    # Filter out excluded sections
    filtered_sections = []
    for section in sections:
        title = section["title"]
        if not title:
            continue
        norm = normalize(title)
        if any(excl in norm for excl in EXCLUDED_SECTIONS):
            continue
        filtered_sections.append(section)
    
    # Combine all non-excluded sections into one text
    combined_text = ""
    for section in filtered_sections:
        section_text = "\n".join(section["content"])
        combined_text += f"\n\n## {section['title']}\n\n{section_text}"
    
    combined_text = combined_text.strip()
    
    # Split the combined text into chunks
    text_chunks = recursive_character_split(combined_text)
    
    print(f"Processing: {md_path}")
    print(f"  Total chunks created: {len(text_chunks)}")
    
    # Process each chunk
    for i, chunk_text in enumerate(text_chunks):
        # Extract tables and images from this chunk
        tables = extract_tables(chunk_text)
        images = extract_images(chunk_text)
        images = fix_image_paths(md_path, images)
        
        # Remove tables from text (but keep the structured table data)
        clean_text = remove_tables(chunk_text)
        
        print(f"  Chunk {i+1}: Tables found: {len(tables)}, Images found: {len(images)}")
        
        chunk = {
            "text": clean_text,
            "tables": tables,
            "images": images
        }
        chunks.append(chunk)
    
    return chunks

def process_all():
    all_chunks = []
    chunk_counter = 1
    for folder in BASE_DIR.iterdir():
        if not folder.is_dir():
            continue
        md_file = folder / f"{folder.name}-referenced.md"
        if md_file.exists():
            chunks = process_md_file(md_file)
            for chunk in chunks:
                chunk_id = f"chunk-{chunk_counter:03d}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "pdf": folder.name,
                    **chunk
                })
                chunk_counter += 1
    return all_chunks

def main():
    chunks = process_all()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(chunks)} chunks to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()