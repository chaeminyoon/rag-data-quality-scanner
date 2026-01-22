"""
CSV data loading and validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union
import uuid

import pandas as pd

from config import get_logger

logger = get_logger("ingest.csv_loader")


@dataclass
class Document:
    """Represents a document with metadata."""

    id: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class QueryGroundTruth:
    """Represents a query with its relevant document IDs."""

    query_id: str
    query: str
    relevant_doc_ids: List[str]
    metadata: Dict = field(default_factory=dict)


class CSVLoader:
    """
    Load documents and ground truth from CSV files.

    Supports common RAG dataset formats with flexible column mapping.
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize CSV loader.

        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding

    def load_documents(
        self,
        file_input: Union[str, Path, BinaryIO],
        text_column: str = "text",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load documents from CSV file.

        Args:
            file_input: File path or file object
            text_column: Name of the text content column
            id_column: Name of the document ID column (auto-generated if None)
            metadata_columns: List of columns to include as metadata

        Returns:
            List of Document objects
        """
        try:
            # Read CSV
            if isinstance(file_input, (str, Path)):
                df = pd.read_csv(file_input, encoding=self.encoding)
                filename = Path(file_input).name
            else:
                df = pd.read_csv(file_input, encoding=self.encoding)
                filename = getattr(file_input, "name", "uploaded.csv")

            logger.info(f"Loading CSV: {filename} ({len(df)} rows)")

            # Validate required column
            if text_column not in df.columns:
                available = ", ".join(df.columns.tolist())
                raise ValueError(
                    f"Text column '{text_column}' not found. Available columns: {available}"
                )

            documents = []
            metadata_columns = metadata_columns or []

            for idx, row in df.iterrows():
                # Get text content
                text = str(row[text_column]).strip()

                if not text or text == "nan":
                    continue

                # Generate or get ID
                if id_column and id_column in df.columns:
                    doc_id = str(row[id_column])
                else:
                    doc_id = f"{filename}_row_{idx}"

                # Collect metadata
                metadata = {
                    "source": filename,
                    "row": idx,
                    "type": "csv",
                }

                for col in metadata_columns:
                    if col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            metadata[col] = value

                documents.append(Document(id=doc_id, text=text, metadata=metadata))

            logger.info(f"Loaded {len(documents)} documents from {filename}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def load_ground_truth(
        self,
        file_input: Union[str, Path, BinaryIO],
        query_column: str = "query",
        relevant_ids_column: str = "relevant_doc_ids",
        query_id_column: Optional[str] = None,
        delimiter: str = ",",
    ) -> List[QueryGroundTruth]:
        """
        Load ground truth queries for evaluation.

        Args:
            file_input: File path or file object
            query_column: Name of the query text column
            relevant_ids_column: Name of the relevant document IDs column
            query_id_column: Name of the query ID column (auto-generated if None)
            delimiter: Delimiter for relevant IDs if stored as string

        Returns:
            List of QueryGroundTruth objects
        """
        try:
            # Read CSV
            if isinstance(file_input, (str, Path)):
                df = pd.read_csv(file_input, encoding=self.encoding)
                filename = Path(file_input).name
            else:
                df = pd.read_csv(file_input, encoding=self.encoding)
                filename = getattr(file_input, "name", "uploaded.csv")

            logger.info(f"Loading ground truth: {filename} ({len(df)} queries)")

            # Validate required columns
            if query_column not in df.columns:
                raise ValueError(f"Query column '{query_column}' not found")
            if relevant_ids_column not in df.columns:
                raise ValueError(f"Relevant IDs column '{relevant_ids_column}' not found")

            ground_truth = []

            for idx, row in df.iterrows():
                query = str(row[query_column]).strip()

                if not query or query == "nan":
                    continue

                # Parse relevant document IDs
                relevant_ids_raw = row[relevant_ids_column]
                if isinstance(relevant_ids_raw, str):
                    relevant_ids = [
                        id.strip() for id in relevant_ids_raw.split(delimiter) if id.strip()
                    ]
                elif isinstance(relevant_ids_raw, list):
                    relevant_ids = [str(id) for id in relevant_ids_raw]
                else:
                    relevant_ids = [str(relevant_ids_raw)]

                # Get or generate query ID
                if query_id_column and query_id_column in df.columns:
                    query_id = str(row[query_id_column])
                else:
                    query_id = f"query_{idx}"

                ground_truth.append(
                    QueryGroundTruth(
                        query_id=query_id,
                        query=query,
                        relevant_doc_ids=relevant_ids,
                        metadata={"source": filename, "row": idx},
                    )
                )

            logger.info(f"Loaded {len(ground_truth)} ground truth queries")
            return ground_truth

        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            raise

    def get_columns(self, file_input: Union[str, Path, BinaryIO]) -> List[str]:
        """
        Get column names from CSV file.

        Args:
            file_input: File path or file object

        Returns:
            List of column names
        """
        if isinstance(file_input, (str, Path)):
            df = pd.read_csv(file_input, nrows=0, encoding=self.encoding)
        else:
            df = pd.read_csv(file_input, nrows=0, encoding=self.encoding)
            file_input.seek(0)  # Reset file pointer

        return df.columns.tolist()

    def preview(
        self, file_input: Union[str, Path, BinaryIO], n_rows: int = 5
    ) -> pd.DataFrame:
        """
        Preview first n rows of CSV file.

        Args:
            file_input: File path or file object
            n_rows: Number of rows to preview

        Returns:
            DataFrame with preview data
        """
        if isinstance(file_input, (str, Path)):
            df = pd.read_csv(file_input, nrows=n_rows, encoding=self.encoding)
        else:
            df = pd.read_csv(file_input, nrows=n_rows, encoding=self.encoding)
            file_input.seek(0)  # Reset file pointer

        return df
