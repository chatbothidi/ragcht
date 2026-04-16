import pickle
from pathlib import Path

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from src.models import DocumentChunk

# Content morpheme tags (nouns, verbs, adjectives)
CONTENT_TAGS = {"NNG", "NNP", "NNB", "VV", "VA", "MAG"}

# Korean stopwords
STOPWORDS = {"하다", "있다", "되다", "이다", "것", "수", "등", "및", "또는", "그", "이", "저"}


class BM25Index:
    def __init__(self, index_dir: str = "./data/bm25_index"):
        self.kiwi = Kiwi()
        self.index_dir = Path(index_dir)
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.chunk_metadata: list[dict] = []

    def tokenize(self, text: str) -> list[str]:
        """Korean morphological tokenization for BM25."""
        tokens = self.kiwi.tokenize(text)
        return [
            token.form
            for token in tokens
            if token.tag in CONTENT_TAGS and token.form not in STOPWORDS
        ]

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index from document chunks."""
        self.chunk_ids = [f"{c.source_file}_{c.chunk_index}" for c in chunks]
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_metadata = [
            {
                "source_file": c.source_file,
                "source_type": c.source_type,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "text": c.text,
            }
            for c in chunks
        ]

        tokenized = [self.tokenize(text) for text in self.chunk_texts]
        if not tokenized:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_type_filter: str | None = None,
    ) -> list[dict]:
        """Search using BM25 scoring."""
        if self.bm25 is None:
            return []

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Create scored results
        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            meta = self.chunk_metadata[idx]
            if source_type_filter and meta["source_type"] != source_type_filter:
                continue
            results.append(
                {
                    "id": self.chunk_ids[idx],
                    "score": float(score),
                    "text": meta["text"],
                    "source_file": meta["source_file"],
                    "source_type": meta["source_type"],
                    "page_number": meta["page_number"],
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self) -> None:
        """Persist BM25 index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_ids": self.chunk_ids,
            "chunk_texts": self.chunk_texts,
            "chunk_metadata": self.chunk_metadata,
            "bm25": self.bm25,
        }
        with open(self.index_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        index_path = self.index_dir / "bm25_index.pkl"
        if not index_path.exists():
            return False

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.chunk_ids = data["chunk_ids"]
        self.chunk_texts = data["chunk_texts"]
        self.chunk_metadata = data["chunk_metadata"]
        self.bm25 = data["bm25"]
        return True
