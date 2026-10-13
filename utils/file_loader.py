
import json
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from utils.logger import logger

SUPPORTED = {".json", ".csv", ".xlsx", ".xls", ".txt", ".md", ".pdf", ".docx"}

def load_file(path: str) -> Tuple[Any, str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    logger.info(f"Loading: {p.name}  ext={ext}")

    if ext == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        sample = json.dumps(
            data[:3] if isinstance(data, list) else data,
            indent=2, default=str, ensure_ascii=False
        )[:4000]
        return data, "json", sample

    elif ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8")
        return df, "csv", df.head(5).to_string()

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
        return df, "excel", df.head(5).to_string()

    elif ext in (".txt", ".md"):
        text = p.read_text(encoding="utf-8")
        return text, "text", text[:4000]

    elif ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(p))
            text = "\n".join(pg.extract_text() or "" for pg in reader.pages)
            return text, "pdf", text[:4000]
        except ImportError:
            raise RuntimeError("Run: pip install PyPDF2")

    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(str(p))
            text = "\n".join(para.text for para in doc.paragraphs if para.text)
            return text, "docx", text[:4000]
        except ImportError:
            raise RuntimeError("Run: pip install python-docx")

    else:
        raise ValueError(f"Unsupported: {ext}. Supported: {sorted(SUPPORTED)}")

def serialize_for_llm(data: Any, fmt: str, max_chars: int = 80_000) -> str:
    if fmt == "json":
        text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    elif fmt in ("csv", "excel"):
        text = data.to_string()
    else:
        text = str(data)
    return text[:max_chars]

def chunk_data(data: Any, fmt: str, chunk_size: int = 2000) -> List[str]:
    if fmt == "json" and isinstance(data, list):
        chunks, buf, buf_len = [], [], 0
        for item in data:
            s = json.dumps(item, default=str, ensure_ascii=False)
            if buf_len + len(s) > chunk_size and buf:
                chunks.append(json.dumps(buf, default=str, ensure_ascii=False))
                buf, buf_len = [], 0
            buf.append(item)
            buf_len += len(s)
        if buf:
            chunks.append(json.dumps(buf, default=str, ensure_ascii=False))
        return chunks
    elif fmt in ("csv", "excel"):
        rpc = max(1, chunk_size // 200)
        return [data.iloc[i:i+rpc].to_string() for i in range(0, len(data), rpc)]
    else:
        text = str(data)
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def smart_sample_for_rag(
    data: Any,
    fmt: str,
    max_chunks: int = 120,
    chunk_size: int = 2000,
) -> List[str]:
    if fmt == "json" and isinstance(data, list):
        total = len(data)
        if total <= max_chunks * 3:
            sampled = data
        else:
            step = total / (max_chunks * 3)
            sampled = [data[int(i * step)] for i in range(max_chunks * 3)]
            logger.info(
                f"[LOADER] Large dataset ({total} records) — "
                f"stratified sample of {len(sampled)} for FAISS."
            )

        chunks = []
        for i in range(0, len(sampled), 3):
            batch = sampled[i:i+3]
            chunk = json.dumps(batch, default=str, ensure_ascii=False)[:chunk_size]
            chunks.append(chunk)
        return chunks

    chunks = chunk_data(data, fmt, chunk_size)
    if len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    return [chunks[int(i * step)] for i in range(max_chunks)]

def select_diverse_examples(data: Any, fmt: str, n: int = 5) -> List[Any]:
    if fmt == "json" and isinstance(data, list):
        total = len(data)
        if total <= n:
            return list(data)
        step = total / n
        return [data[int(i * step)] for i in range(n)]
    return []
