import hashlib
import pickle
from pathlib import Path

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from src.models import DocumentChunk

# 단어종류 (명사, 동사, 형용사)
CONTENT_TAGS = {"NNG", "NNP", "NNB", "VV", "VA", "MAG"}

# 한국어 불용어
STOPWORDS = {"하다", "있다", "되다", "이다", "것", "수", "등", "및", "또는", "그", "이", "저"}

class BM25Index:
    def __init__(self, index_dir: str = "./data/bm25_index"):
        self.kiwi = Kiwi()
        self.index_dir = Path(index_dir)
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.chunk_metadata: list[dict] = []
        # 텍스트 hash -> 토큰 리스트. 변경 없는 청크는 재토큰화 스킵.
        self._token_cache: dict[str, list[str]] = {}

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
                "post_id": (c.metadata or {}).get("post_id"),
                "post_title": (c.metadata or {}).get("post_title"),
                "year": (c.metadata or {}).get("year"),
                "url": (c.metadata or {}).get("url"),
            }
            for c in chunks
        ]

        # 토큰화 캐시 사용 — 같은 텍스트는 다시 Kiwi에 돌리지 않음.
        # 새 캐시를 만들면서 이번 build에 등장한 키만 살리므로 삭제된 청크는 자동 prune.
        new_cache: dict[str, list[str]] = {}
        tokenized: list[list[str]] = []
        for text in self.chunk_texts:
            key = hashlib.md5(text.encode("utf-8")).hexdigest()
            tokens = new_cache.get(key) or self._token_cache.get(key)
            if tokens is None:
                tokens = self.tokenize(text)
            new_cache[key] = tokens
            tokenized.append(tokens)
        self._token_cache = new_cache

        if not tokenized:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_type_filter: str | None = None,
        year_filter: int | None = None,
    ) -> list[dict]:
        """Search using BM25 scoring."""
        if self.bm25 is None:
            return []

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 점수 계산 결과
        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            meta = self.chunk_metadata[idx]
            if source_type_filter and meta["source_type"] != source_type_filter:
                continue
            if year_filter is not None and meta.get("year") != year_filter:
                continue
            results.append(
                {
                    "id": self.chunk_ids[idx],
                    "score": float(score),
                    "text": meta["text"],
                    "source_file": meta["source_file"],
                    "source_type": meta["source_type"],
                    "page_number": meta["page_number"],
                    "post_id": meta.get("post_id"),
                    "post_title": meta.get("post_title"),
                    "year": meta.get("year"),
                    "url": meta.get("url"),
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
            "token_cache": self._token_cache,
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
        # 이전 버전 pkl과의 호환을 위해 get으로 안전하게 로드.
        self._token_cache = data.get("token_cache", {})
        return True
