"""
Pinecone Serverless client for vector operations.
"""

import time
import uuid
from typing import Callable, Dict, List, Optional

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_logger, get_settings

logger = get_logger("vectordb.pinecone_client")


class PineconeClient:
    """
    Pinecone Serverless client for vector operations.

    Features:
    - Serverless index creation
    - Namespace-based data isolation (for A/B comparison)
    - Batch upsert with progress tracking
    - Automatic cleanup
    """

    UPSERT_BATCH_SIZE = 100

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        """
        Initialize Pinecone client.

        Args:
            api_key: Pinecone API key (uses settings if not provided)
            environment: Pinecone environment/region
            index_name: Default index name
        """
        settings = get_settings()

        self.api_key = api_key or settings.PINECONE_API_KEY.get_secret_value()
        self.environment = environment or settings.PINECONE_ENVIRONMENT
        self.index_name = index_name or settings.PINECONE_INDEX_NAME

        self.pc = Pinecone(api_key=self.api_key)
        self._index = None

        logger.info(f"Initialized Pinecone client for {self.environment}")

    def create_index(
        self,
        name: Optional[str] = None,
        dimension: int = 1024,
        metric: str = "cosine",
    ) -> None:
        """
        Create a serverless index if it doesn't exist.

        Args:
            name: Index name (uses default if not provided)
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        name = name or self.index_name

        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if name not in existing_indexes:
            logger.info(f"Creating new index: {name} (dimension={dimension}, metric={metric})")

            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment,
                ),
            )

            # Wait for index to be ready
            while not self.pc.describe_index(name).status.get("ready", False):
                logger.info("Waiting for index to be ready...")
                time.sleep(1)

            logger.info(f"Index {name} created successfully")
        else:
            logger.info(f"Using existing index: {name}")

        self._index = self.pc.Index(name)

    def get_index(self, name: Optional[str] = None):
        """
        Get index instance.

        Args:
            name: Index name

        Returns:
            Pinecone Index object
        """
        name = name or self.index_name

        if self._index is None:
            self._index = self.pc.Index(name)

        return self._index

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def upsert_documents(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        namespace: str = "default",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict:
        """
        Upsert documents with embeddings.

        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
            embeddings: np.ndarray of shape (n_docs, dimension)
            namespace: Namespace for data isolation
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with upsert statistics
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) count mismatch")

        index = self.get_index()

        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.get("id", str(uuid.uuid4()))
            metadata = {
                "text": doc["text"][:1000],  # Truncate for metadata limit
                **doc.get("metadata", {}),
            }

            vectors.append({
                "id": doc_id,
                "values": embedding.tolist(),
                "metadata": metadata,
            })

        logger.info(f"Upserting {len(vectors)} vectors to namespace '{namespace}'")

        # Batch upsert
        total_upserted = 0
        for i in range(0, len(vectors), self.UPSERT_BATCH_SIZE):
            batch = vectors[i : i + self.UPSERT_BATCH_SIZE]
            index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)

            if progress_callback:
                progress_callback(total_upserted / len(vectors))

        logger.info(f"Successfully upserted {total_upserted} vectors")
        return {"upserted_count": total_upserted}

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        namespace: str = "default",
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Query similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            namespace: Namespace to query
            filter: Optional metadata filter
            include_metadata: Include metadata in results

        Returns:
            List of matches with id, score, and metadata
        """
        index = self.get_index()

        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )

        matches = []
        for match in results.matches:
            matches.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else None,
            })

        return matches

    def query_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        namespace: str = "default",
        filter: Optional[Dict] = None,
    ) -> List[List[Dict]]:
        """
        Query multiple vectors in batch.

        Args:
            query_embeddings: Array of query vectors
            top_k: Number of results per query
            namespace: Namespace to query
            filter: Optional metadata filter

        Returns:
            List of result lists
        """
        results = []
        for embedding in query_embeddings:
            matches = self.query(
                query_embedding=embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
            )
            results.append(matches)

        return results

    def delete_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in a namespace.

        Args:
            namespace: Namespace to delete
        """
        index = self.get_index()

        try:
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace: {namespace}")
        except Exception as e:
            logger.warning(f"Failed to delete namespace {namespace}: {e}")

    def delete_index(self, name: Optional[str] = None) -> None:
        """
        Delete entire index.

        Args:
            name: Index name
        """
        name = name or self.index_name

        try:
            self.pc.delete_index(name)
            logger.info(f"Deleted index: {name}")
            self._index = None
        except Exception as e:
            logger.warning(f"Failed to delete index {name}: {e}")

    def get_stats(self, namespace: Optional[str] = None) -> Dict:
        """
        Get index statistics.

        Args:
            namespace: Optional namespace filter

        Returns:
            Index statistics
        """
        index = self.get_index()
        stats = index.describe_index_stats()

        if namespace:
            ns_stats = stats.namespaces.get(namespace, {})
            return {
                "namespace": namespace,
                "vector_count": ns_stats.get("vector_count", 0),
            }

        return {
            "total_vector_count": stats.total_vector_count,
            "namespaces": {
                ns: {"vector_count": ns_info.vector_count}
                for ns, ns_info in stats.namespaces.items()
            },
        }

    def validate_connection(self) -> bool:
        """
        Validate API connection is working.

        Returns:
            True if connection is valid
        """
        try:
            self.pc.list_indexes()
            logger.info("Pinecone connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"Pinecone connection validation failed: {e}")
            return False
