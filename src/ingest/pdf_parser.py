"""
PDF text extraction using PyMuPDF (fitz).
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

import fitz  # PyMuPDF

from config import get_logger

logger = get_logger("ingest.pdf_parser")


@dataclass
class Document:
    """Represents a document chunk with metadata."""

    id: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())


class PDFParser:
    """
    Extract text from PDF documents.

    Features:
    - Multi-page extraction
    - Page-level metadata preservation
    - Configurable text cleaning
    """

    def __init__(self, min_text_length: int = 10):
        """
        Initialize PDF parser.

        Args:
            min_text_length: Minimum text length to include a page
        """
        self.min_text_length = min_text_length

    def parse(
        self,
        file_input: Union[str, Path, BinaryIO, bytes],
        filename: Optional[str] = None,
    ) -> List[Document]:
        """
        Extract text from PDF file.

        Args:
            file_input: File path, file object, or bytes
            filename: Optional filename for metadata

        Returns:
            List of Document objects, one per page with content
        """
        documents = []

        try:
            # Handle different input types
            if isinstance(file_input, (str, Path)):
                doc = fitz.open(str(file_input))
                filename = filename or Path(file_input).name
            elif isinstance(file_input, bytes):
                doc = fitz.open(stream=file_input, filetype="pdf")
            else:
                # File-like object (BytesIO, UploadedFile)
                content = file_input.read()
                if hasattr(file_input, "name"):
                    filename = filename or file_input.name
                doc = fitz.open(stream=content, filetype="pdf")

            filename = filename or "unknown.pdf"
            total_pages = len(doc)

            logger.info(f"Parsing PDF: {filename} ({total_pages} pages)")

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text")

                # Clean and validate text
                text = self._clean_text(text)

                if len(text) >= self.min_text_length:
                    doc_id = f"{filename}_page_{page_num + 1}"
                    documents.append(
                        Document(
                            id=doc_id,
                            text=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "total_pages": total_pages,
                                "type": "pdf",
                            },
                        )
                    )

            doc.close()
            logger.info(f"Extracted {len(documents)} documents from {filename}")

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise

        return documents

    def parse_to_single_document(
        self,
        file_input: Union[str, Path, BinaryIO, bytes],
        filename: Optional[str] = None,
    ) -> Document:
        """
        Extract all text from PDF as a single document.

        Args:
            file_input: File path, file object, or bytes
            filename: Optional filename for metadata

        Returns:
            Single Document with all text concatenated
        """
        documents = self.parse(file_input, filename)

        if not documents:
            return Document(id="empty", text="", metadata={"source": filename or "unknown.pdf"})

        # Combine all text
        combined_text = "\n\n".join(doc.text for doc in documents)

        return Document(
            id=documents[0].metadata.get("source", "combined"),
            text=combined_text,
            metadata={
                "source": documents[0].metadata.get("source", "unknown.pdf"),
                "total_pages": documents[0].metadata.get("total_pages", len(documents)),
                "type": "pdf",
            },
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)

        text = " ".join(cleaned_lines)

        # Normalize whitespace
        while "  " in text:
            text = text.replace("  ", " ")

        return text.strip()
