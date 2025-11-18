"""Text chunking utilities used by the ingestion pipeline.

The main helper, `chunk_text`, receives a list of paragraphs and splits
them into overlapping chunks of approximately `chunk_size` characters.
"""

from __future__ import annotations

import re
from typing import List, Sequence


def chunk_text(
    paragraphs: Sequence[str],
    *,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """Split paragraphs into overlapping chunks of text.

    The algorithm tries to respect paragraph boundaries. When a single
    paragraph is longer than `chunk_size`, it is further split into
    sentences using a simple punctuation-based heuristic.

    Parameters
    ----------
    paragraphs:
        List of already cleaned paragraphs.
    chunk_size:
        Target maximum length in characters for each chunk.
    overlap:
        Number of trailing characters from the previous chunk that are
        carried over as prefix to the next one, to preserve context.
    """
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        # When a paragraph is too long, split it into sentences first
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[\.\!\?\;:])\s+", para)
            for sent in sentences:
                candidate = (buffer + " " + sent).strip() if buffer else sent
                if len(candidate) <= chunk_size:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer)
                        # Keep only the overlapping tail in the buffer
                        if overlap and len(buffer) > overlap:
                            buffer = buffer[-overlap:]
                        else:
                            buffer = ""
                    buffer = (buffer + " " + sent).strip() if buffer else sent
            continue

        # Normal case: paragraph fits inside the chunk size
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
                if overlap and len(buffer) > overlap:
                    buffer = buffer[-overlap:]
                else:
                    buffer = ""
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

    if buffer:
        chunks.append(buffer)

    # Remove any empty or whitespace-only chunks
    return [c.strip() for c in chunks if c.strip()]
