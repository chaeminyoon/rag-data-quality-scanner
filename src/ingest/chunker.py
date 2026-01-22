"""
Text chunking strategies for optimal retrieval.
"""

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config import get_logger

logger = get_logger("ingest.chunker")


class ChunkStrategy(Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class Document:
    """Represents a document chunk with metadata."""

    id: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class TextChunker:
    """
    Chunk documents for optimal retrieval.

    Supports multiple chunking strategies with configurable
    chunk sizes and overlap.
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    # Paragraph patterns
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to include
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(
        self,
        documents: List[Document],
        strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
    ) -> List[Document]:
        """
        Chunk documents using specified strategy.

        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy to use

        Returns:
            List of chunked documents
        """
        chunked_documents = []

        for doc in documents:
            if strategy == ChunkStrategy.FIXED_SIZE:
                chunks = self._chunk_fixed_size(doc.text)
            elif strategy == ChunkStrategy.SENTENCE:
                chunks = self._chunk_by_sentence(doc.text)
            elif strategy == ChunkStrategy.PARAGRAPH:
                chunks = self._chunk_by_paragraph(doc.text)
            else:
                chunks = self._chunk_fixed_size(doc.text)

            for i, chunk_text in enumerate(chunks):
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_id = f"{doc.id}_chunk_{i}"
                    chunk_metadata = {
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "parent_id": doc.id,
                        "chunk_strategy": strategy.value,
                    }
                    chunked_documents.append(
                        Document(id=chunk_id, text=chunk_text, metadata=chunk_metadata)
                    )

        logger.info(
            f"Chunked {len(documents)} documents into {len(chunked_documents)} chunks "
            f"using {strategy.value} strategy"
        )

        return chunked_documents

    def _chunk_fixed_size(self, text: str) -> List[str]:
        """
        Chunk text by fixed character size with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to find a good break point
            if end < len(text):
                # Look for whitespace to break at
                break_point = text.rfind(" ", start + self.min_chunk_size, end)
                if break_point > start:
                    end = break_point

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[str]:
        """
        Chunk text by sentences, combining until chunk_size is reached.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If single sentence exceeds chunk size, use fixed-size chunking
            if sentence_length > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Chunk the long sentence
                sub_chunks = self._chunk_fixed_size(sentence)
                chunks.extend(sub_chunks)
                continue

            # Check if adding sentence would exceed chunk size
            if current_length + sentence_length + 1 > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap (last sentence)
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + sentence_length + 1
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """
        Chunk text by paragraphs, combining small paragraphs.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # Split into paragraphs
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # If single paragraph exceeds chunk size, use sentence chunking
            if para_length > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Chunk the long paragraph by sentences
                sub_chunks = self._chunk_by_sentence(para)
                chunks.extend(sub_chunks)
                continue

            # Check if adding paragraph would exceed chunk size
            if current_length + para_length + 2 > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 2

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def estimate_chunks(
        self, documents: List[Document], strategy: ChunkStrategy = ChunkStrategy.SENTENCE
    ) -> int:
        """
        Estimate number of chunks without actually chunking.

        Args:
            documents: List of documents
            strategy: Chunking strategy

        Returns:
            Estimated number of chunks
        """
        total_chars = sum(len(doc.text) for doc in documents)
        avg_chunk_size = self.chunk_size - self.chunk_overlap

        return max(1, total_chars // avg_chunk_size)
