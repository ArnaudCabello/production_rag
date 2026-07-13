"""Build the distributable desktop app with PyInstaller.

Run from the repo root, after building the frontend (npm run build):
    pip install pyinstaller
    python packaging/build.py

Produces dist/production-rag/ (onedir — starts fast; onefile would re-extract
gigabytes on every launch) and zips it as production-rag-<os>.zip. Executables
are per-OS: run this on (or CI for) each platform you distribute to — see
.github/workflows/release.yml.
"""
import os
import platform
import shutil
import sys
from pathlib import Path

import PyInstaller.__main__

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST = REPO_ROOT / "frontend" / "dist"

# Packages with dynamic imports, plugin registries, or bundled data files that
# PyInstaller's static analysis misses. collect-all trades bundle size for
# not debugging one missing module per build.
COLLECT_ALL = [
    "docling", "docling_core", "docling_ibm_models", "docling_parse",
    "chromadb", "onnxruntime", "tokenizers", "safetensors",
    "transformers", "sentence_transformers", "huggingface_hub",
    "langchain", "langchain_core", "langgraph",
    "langchain_anthropic", "langchain_openai", "langchain_google_genai",
    "uvicorn", "keyring", "platformdirs", "rank_bm25", "pypdfium2",
]

# GPU-only or unused packages that must not ride along (GitHub Releases cap
# assets at 2GB, so bundle size is a hard constraint). The packaged app
# generates through API providers; local-GPU generation stays a dev feature.
# Build with CPU torch (see the workflow) or the nvidia/triton libs come too.
EXCLUDES = ["bitsandbytes", "triton", "faiss", "gradio"]


def main():
    os.chdir(REPO_ROOT)
    if not DIST.is_dir():
        sys.exit("frontend/dist missing — run `npm run build` in frontend/ first")

    args = [
        "desktop.py",
        "--name", "production-rag",
        "--onedir", "--noconfirm", "--clean",
        "--add-data", f"{DIST}{os.pathsep}frontend/dist",
    ]
    for pkg in COLLECT_ALL:
        args += ["--collect-all", pkg]
    for pkg in EXCLUDES:
        args += ["--exclude-module", pkg]

    PyInstaller.__main__.run(args)

    archive = f"production-rag-{platform.system().lower()}-{platform.machine().lower()}"
    shutil.make_archive(archive, "zip", REPO_ROOT / "dist" / "production-rag")
    print(f"\nBuilt dist/production-rag/ and {archive}.zip")


if __name__ == "__main__":
    main()
