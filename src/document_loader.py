from pathlib import Path

import chardet
import fitz  # pymupdf
from docx import Document as DocxDocument
from google.cloud import documentai_v1 as documentai

from src.models import LoadedDocument

# Special unicode characters that cause markdown rendering issues
_NORMALIZE_MAP = str.maketrans({
    "\u223c": "~",   # TILDE OPERATOR → ~
    "\u301c": "~",   # WAVE DASH → ~
    "\uff5e": "~",   # FULLWIDTH TILDE → ~
    "\u2212": "-",   # MINUS SIGN → -
    "\u2013": "-",   # EN DASH → -
    "\u2014": "-",   # EM DASH → -
    "\u00a0": " ",   # NO-BREAK SPACE → space
})


def _normalize_text(text: str) -> str:
    """Normalize special unicode characters from PDF extraction."""
    return text.translate(_NORMALIZE_MAP)


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

    # Document AI OCR processor
    DOCAI_PROCESSOR = "projects/544100649926/locations/us/processors/f0b50f9a399a78e6"

    def __init__(self):
        self._docai_client = None

    def load_file(self, file_path: str, source_type: str) -> LoadedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf(path, source_type)
        elif ext in (".docx", ".doc"):
            return self._load_docx(path, source_type)
        elif ext in (".txt", ".md"):
            return self._load_text(path, source_type)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def load_directory(self, directory: str) -> list[LoadedDocument]:
        docs = []
        base_path = Path(directory)

        for subdir in ["medical", "events"]:
            subdir_path = base_path / subdir
            if not subdir_path.exists():
                continue

            source_type = "medical" if subdir == "medical" else "event"

            for file_path in subdir_path.rglob("*"):
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    doc = self.load_file(str(file_path), source_type)
                    docs.append(doc)

        return docs

    def _load_pdf(self, path: Path, source_type: str) -> LoadedDocument:
        # First try pymupdf text extraction
        doc = fitz.open(str(path))
        pages = []
        full_text_parts = []
        has_text = False

        for page_num, page in enumerate(doc, start=1):
            text = _normalize_text(page.get_text().strip())
            if text:
                has_text = True
            pages.append({"page": page_num, "text": text})
            full_text_parts.append(text)

        doc.close()

        # If no text extracted, fall back to Document AI OCR
        if not has_text:
            print(f"  [OCR] No text in {path.name}, using Document AI OCR...")
            return self._load_pdf_with_ocr(path, source_type)

        return LoadedDocument(
            text="\n\n".join(full_text_parts),
            source_file=path.name,
            source_type=source_type,
            pages=pages,
            metadata={"total_pages": len(pages)},
        )

    def _load_pdf_with_ocr(self, path: Path, source_type: str) -> LoadedDocument:
        """Extract text from scanned PDF using Google Document AI OCR."""
        if self._docai_client is None:
            self._docai_client = documentai.DocumentProcessorServiceClient(
                client_options={"api_endpoint": "us-documentai.googleapis.com"}
            )

        raw_document = documentai.RawDocument(
            content=path.read_bytes(),
            mime_type="application/pdf",
        )

        request = documentai.ProcessRequest(
            name=self.DOCAI_PROCESSOR,
            raw_document=raw_document,
        )

        result = self._docai_client.process_document(request=request)
        document = result.document

        # Extract text per page
        pages = []
        for i, page in enumerate(document.pages, start=1):
            page_text = self._extract_page_text(document.text, page)
            pages.append({"page": i, "text": page_text})

        full_text = _normalize_text(document.text)

        return LoadedDocument(
            text=full_text,
            source_file=path.name,
            source_type=source_type,
            pages=pages,
            metadata={"total_pages": len(pages), "ocr": True},
        )

    def _extract_page_text(self, full_text: str, page) -> str:
        """Extract text for a specific page from Document AI result."""
        segments = []
        for block in page.blocks:
            for paragraph in block.layout.text_anchor.text_segments:
                start = int(paragraph.start_index) if paragraph.start_index else 0
                end = int(paragraph.end_index)
                segments.append(full_text[start:end])
        return "".join(segments)

    def _load_docx(self, path: Path, source_type: str) -> LoadedDocument:
        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        return LoadedDocument(
            text="\n\n".join(paragraphs),
            source_file=path.name,
            source_type=source_type,
            metadata={"paragraph_count": len(paragraphs)},
        )

    def _load_text(self, path: Path, source_type: str) -> LoadedDocument:
        raw = path.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        text = raw.decode(encoding)

        return LoadedDocument(
            text=text,
            source_file=path.name,
            source_type=source_type,
        )
