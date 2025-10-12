# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import List


def chunk_text(paragraphs: List[str], *, chunk_size: int, overlap: int) -> List[str]:
    """Chunking por párrafos y partición por oraciones si un párrafo es demasiado largo."""
    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[\.\!\?\;:])\s+", para)
            for sent in sentences:
                candidate = (buffer + " " + sent).strip() if buffer else sent
                if len(candidate) <= chunk_size:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer)
                        buffer = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
                    buffer = (buffer + " " + sent).strip() if buffer else sent
            continue
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
                buffer = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
    if buffer:
        chunks.append(buffer)
    return [c.strip() for c in chunks if c.strip()]
