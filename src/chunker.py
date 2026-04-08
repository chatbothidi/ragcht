import re

from kiwipiepy import Kiwi
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.models import DocumentChunk, LoadedDocument


class DocumentChunker:
    # Common Korean medical document section headers
    MEDICAL_HEADERS = re.compile(
        r"^(진단|소견|처방|검사결과|주소|병력|현병력|과거력|가족력|"
        r"신체검사|치료계획|경과|수술소견|퇴원요약|간호기록|"
        r"투약|주의사항|부작용|용법|효능|성분)[\s:：]",
        re.MULTILINE,
    )

    def __init__(self, settings: Settings):
        self.kiwi = Kiwi()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        self.event_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=300,
            separators=["\n\n", "\n", "。", ".", "! ", "? ", ", ", " "],
        )

    def chunk_document(self, document: LoadedDocument) -> list[DocumentChunk]:
        if document.source_type == "medical":
            return self._chunk_medical(document)
        else:
            return self._chunk_event(document)

    def chunk_documents(self, documents: list[LoadedDocument]) -> list[DocumentChunk]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _chunk_medical(self, document: LoadedDocument) -> list[DocumentChunk]:
        """Hierarchical chunking: section headers → sentence boundaries."""
        sections = self._split_by_sections(document.text)
        chunks = []

        for section_title, section_text in sections:
            sentences = self._split_sentences(section_text)
            section_chunks = self._group_sentences(
                sentences,
                max_chars=self.chunk_size * 2,  # ~1.5 chars per token
                overlap_chars=self.chunk_overlap * 2,
            )

            for i, chunk_text in enumerate(section_chunks):
                text = chunk_text
                if section_title:
                    text = f"[{section_title}] {chunk_text}"

                chunks.append(
                    DocumentChunk(
                        text=text,
                        source_file=document.source_file,
                        source_type=document.source_type,
                        chunk_index=len(chunks),
                        section_title=section_title,
                        page_number=self._estimate_page(document, chunk_text),
                    )
                )

        return chunks

    def _chunk_event(self, document: LoadedDocument) -> list[DocumentChunk]:
        """Standard recursive splitting for event documents."""
        texts = self.event_splitter.split_text(document.text)

        return [
            DocumentChunk(
                text=text,
                source_file=document.source_file,
                source_type=document.source_type,
                chunk_index=i,
                page_number=self._estimate_page(document, text),
            )
            for i, text in enumerate(texts)
        ]

    def _split_by_sections(self, text: str) -> list[tuple[str | None, str]]:
        """Split text by medical section headers."""
        matches = list(self.MEDICAL_HEADERS.finditer(text))

        if not matches:
            return [(None, text)]

        sections = []

        # Text before first header
        if matches[0].start() > 0:
            sections.append((None, text[: matches[0].start()].strip()))

        for i, match in enumerate(matches):
            title = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((title, section_text))

        return sections

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using kiwipiepy."""
        result = self.kiwi.split_into_sents(text)
        return [sent.text.strip() for sent in result if sent.text.strip()]

    def _group_sentences(
        self,
        sentences: list[str],
        max_chars: int,
        overlap_chars: int,
    ) -> list[str]:
        """Group sentences into chunks with overlap at sentence boundaries."""
        if not sentences:
            return []

        chunks = []
        current_sentences: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_len + sent_len > max_chars and current_sentences:
                chunks.append(" ".join(current_sentences))

                # Calculate overlap: keep last sentences within overlap_chars
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) > overlap_chars:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)

                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _estimate_page(
        self, document: LoadedDocument, chunk_text: str
    ) -> int | None:
        """Estimate page number for a chunk based on PDF pages."""
        if not document.pages:
            return None

        chunk_start = chunk_text[:100]
        for page_info in document.pages:
            if chunk_start in page_info["text"]:
                return page_info["page"]

        return None
