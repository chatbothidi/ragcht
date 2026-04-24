import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import chardet
import fitz  # pymupdf
fitz.TOOLS.mupdf_display_errors(False)
from docx import Document as DocxDocument
from google.cloud import documentai_v1 as documentai
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

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


def _normalize_table_separators(text: str) -> str:
    """Normalize overly long markdown table separators.

    Converts patterns like '|:----...----:|' (300+ chars) to '|:---:|'.
    """
    import re
    # Match sequences of dashes (10+) optionally surrounded by colons
    return re.sub(r":?-{10,}:?", lambda m: ":---:" if ":" in m.group() else "---", text)


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    # Document AI OCR processor
    DOCAI_PROCESSOR = "projects/544100649926/locations/us/processors/f0b50f9a399a78e6"

    CACHE_FILENAME = ".image_cache.json"

    def __init__(self, genai_client=None):
        self._docai_client = None
        self._genai_client = genai_client
        self._current_cache: dict = {}
        self._current_cache_path: Path | None = None
        self._cache_dirty = False

    def _load_image_cache(self, post_dir: Path) -> dict:
        """Load image cache for a post directory."""
        cache_path = post_dir / self.CACHE_FILENAME
        self._current_cache_path = cache_path
        self._cache_dirty = False
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                self._current_cache = json.load(f)
        else:
            self._current_cache = {}
        return self._current_cache

    def _save_image_cache(self) -> None:
        """Save image cache if modified."""
        if self._cache_dirty and self._current_cache_path:
            with open(self._current_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._current_cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """Compute MD5 hash of binary data."""
        return hashlib.md5(data).hexdigest()

    def _get_cached_text(self, key: str, data: bytes) -> str | None:
        """Check cache for extracted text. Returns cached text or None."""
        file_hash = self._compute_hash(data)
        cached = self._current_cache.get(key)
        if cached and cached.get("hash") == file_hash:
            print(f"  [Cache hit] {key}")
            return cached["extracted_text"]
        return None

    def _set_cache(self, key: str, data: bytes, text: str) -> None:
        """Save extracted text to cache."""
        self._current_cache[key] = {
            "hash": self._compute_hash(data),
            "extracted_text": text,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cache_dirty = True

    def load_file(self, file_path: str, source_type: str, post_id: int | None = None,
                  post_title: str | None = None, attachments: list[str] | None = None,
                  year: int | None = None) -> LoadedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()

        fitz.TOOLS.mupdf_warnings(reset=True)

        if ext in self.IMAGE_EXTENSIONS:
            doc = self._load_image(path, source_type)
        elif ext == ".pdf":
            doc = self._load_pdf(path, source_type)
        elif ext in (".docx", ".doc"):
            doc = self._load_docx(path, source_type)
        elif ext in (".txt", ".md"):
            doc = self._load_text(path, source_type)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        warnings = fitz.TOOLS.mupdf_warnings()
        if warnings:
            print(f"    [MuPDF WARNING] {path.name}:\n      {warnings.replace(chr(10), chr(10) + '      ')}")

        # Add post metadata
        if doc.metadata is None:
            doc.metadata = {}
        if post_id is not None:
            doc.metadata["post_id"] = post_id
        if post_title:
            doc.metadata["post_title"] = post_title
        if attachments:
            doc.metadata["attachments"] = attachments
        if year is not None:
            doc.metadata["year"] = year

        return doc

    def load_directory(self, directory: str) -> list[LoadedDocument]:
        """Load documents from post ID-based directory structure."""
        docs = []
        base_path = Path(directory)

        for post_dir in sorted(base_path.iterdir()):
            if not post_dir.is_dir():
                continue

            # Read data.json for metadata
            data_json = post_dir / "data.json"
            if not data_json.exists():
                # Fallback: old structure (medical/events folders)
                if post_dir.name in ("medical", "events"):
                    docs.extend(self._load_old_structure(post_dir))
                continue

            with open(data_json, encoding="utf-8") as f:
                metadata = json.load(f)

            # Load image cache for this post
            self._load_image_cache(post_dir)

            post_id = metadata.get("id")
            category = metadata.get("category", "unknown")
            title = metadata.get("title", "")
            main_file = metadata.get("main_file", "")
            attachments = metadata.get("attachments", [])
            year = metadata.get("year")

            all_supported = self.SUPPORTED_EXTENSIONS | self.IMAGE_EXTENSIONS

            # Load main file
            main_path = post_dir / main_file
            if main_path.exists() and main_path.suffix.lower() in all_supported:
                doc = self.load_file(
                    str(main_path), category,
                    post_id=post_id, post_title=title,
                    attachments=attachments, year=year,
                )
                docs.append(doc)

            # Load attachment files
            for att_name in attachments:
                att_path = post_dir / att_name
                if att_path.exists() and att_path.suffix.lower() in all_supported:
                    doc = self.load_file(
                        str(att_path), category,
                        post_id=post_id, post_title=title, year=year,
                    )
                    docs.append(doc)

            # Save image cache for this post
            self._save_image_cache()

        return docs

    def _load_old_structure(self, subdir_path: Path) -> list[LoadedDocument]:
        """Fallback: load from old medical/events folder structure."""
        docs = []
        source_type = "medical" if subdir_path.name == "medical" else "event"
        for file_path in subdir_path.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self.load_file(str(file_path), source_type)
                docs.append(doc)
        return docs

    @staticmethod
    def _is_low_quality_text(text: str) -> bool:
        """Check the quality of texts extracted pdfs if it is good or not."""
        if not text.strip():
            return True

        chars = len(text.replace(" ", "").replace("\n", ""))
        total = len(text)

        if chars == 0:
            return True

        # Portion of blank is over 50% (Bad)
        if 1 - (chars / total) > 0.5:
            return True

        # Ａverage length of words is lower than 2 (Bad)
        words = text.split()
        if words and sum(len(w) for w in words) / len(words) < 2:
            return True

        return False

    def _load_pdf(self, path: Path, source_type: str) -> LoadedDocument:
        pdf_data = path.read_bytes()

        # Check cache
        cached_text = self._get_cached_text(path.name, pdf_data)
        if cached_text is not None:
            return LoadedDocument(
                text=cached_text,
                source_file=path.name,
                source_type=source_type,
                metadata={"ocr": True, "cached": True},
            )

        # Cache miss - run Document AI OCR
        if self._docai_client is None:
            self._docai_client = documentai.DocumentProcessorServiceClient(
                client_options={"api_endpoint": "us-documentai.googleapis.com"}
            )

        # Check page count - Document AI online limit is 30 pages
        doc = fitz.open(str(path))
        total_pages = len(doc)
        doc.close()

        if total_pages <= 15:
            print(f"  [OCR] Processing {path.name} ({total_pages} pages)...")
            full_text, pages = self._ocr_pdf_bytes(pdf_data)
        else:
            print(f"  [OCR] Processing {path.name} ({total_pages} pages, chunked)...")
            full_text, pages = self._ocr_pdf_chunked(path, total_pages)

        full_text = _normalize_text(full_text)

        # Save to cache
        self._set_cache(path.name, pdf_data, full_text)

        return LoadedDocument(
            text=full_text,
            source_file=path.name,
            source_type=source_type,
            pages=pages,
            metadata={"total_pages": len(pages), "ocr": True},
        )

    def _ocr_pdf_bytes(self, pdf_data: bytes) -> tuple[str, list[dict]]:
        """OCR a PDF that fits within the 30-page limit."""
        raw_document = documentai.RawDocument(
            content=pdf_data,
            mime_type="application/pdf",
        )
        request = documentai.ProcessRequest(
            name=self.DOCAI_PROCESSOR,
            raw_document=raw_document,
        )
        result = self._docai_client.process_document(request=request)
        document = result.document

        pages = []
        for i, page in enumerate(document.pages, start=1):
            page_text = self._extract_page_text(document.text, page)
            pages.append({"page": i, "text": page_text})

        return document.text, pages

    def _ocr_pdf_chunked(self, path: Path, total_pages: int, chunk_size: int = 15) -> tuple[str, list[dict]]:
        """OCR a large PDF by splitting into chunks of 30 pages."""
        all_text_parts = []
        all_pages = []

        doc = fitz.open(str(path))

        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            print(f"    Pages {start + 1}-{end}...")

            # Extract page range to temporary PDF
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
            chunk_bytes = chunk_doc.tobytes()
            chunk_doc.close()

            # OCR this chunk
            text, pages = self._ocr_pdf_bytes(chunk_bytes)

            all_text_parts.append(text)
            # Adjust page numbers
            for page in pages:
                page["page"] += start
            all_pages.extend(pages)

        doc.close()

        return "\n\n".join(all_text_parts), all_pages

    def _extract_page_text(self, full_text: str, page) -> str:
        segments = []
        for block in page.blocks:
            for paragraph in block.layout.text_anchor.text_segments:
                start = int(paragraph.start_index) if paragraph.start_index else 0
                end = int(paragraph.end_index)
                segments.append(full_text[start:end])
        return "".join(segments)

    def _load_image(self, path: Path, source_type: str) -> LoadedDocument:
        """Extract text from image file using Gemini Vision, with caching."""
        img_data = path.read_bytes()

        # Check cache
        cached_text = self._get_cached_text(path.name, img_data)
        if cached_text is not None:
            return LoadedDocument(
                text=cached_text, source_file=path.name, source_type=source_type,
                metadata={"image_ocr": True, "cached": True},
            )

        if self._genai_client is None:
            from src.config import get_settings
            settings = get_settings()
            self._genai_client = genai.Client(
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.llm_location or settings.gcp_location,
            )
            self._llm_model = settings.llm_model

        print(f"  [Vision] Extracting text from image: {path.name}")

        ext = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
        mime_type = mime_map.get(ext, "image/jpeg")

        response = self._genai_client.models.generate_content(
            model=self._llm_model,
            contents=Content(
                role="user",
                parts=[
                    Part(inline_data={"mime_type": mime_type, "data": base64.b64encode(img_data).decode()}),
                    Part(text="""이 이미지에서 텍스트를 추출해 주세요. 다음 규칙을 따르세요:
1. 표(시간표, 프로그램 등)는 간결한 마크다운 표로 변환하세요. 구분선은 반드시 '| --- |' 형태로 짧게 유지하고, 절대 '-' 문자를 10개 이상 반복하지 마세요. 셀 내용에 과도한 공백(padding)을 넣지 마세요.
2. 일반 텍스트(제목, 장소, 안내문 등)는 그대로 텍스트로 출력하세요.
3. 표와 텍스트를 구분하여 정리하세요."""),
                ],
            ),
            config=GenerateContentConfig(
                max_output_tokens=8192,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )

        text = response.text or ""
        if not text:
            print(f"  [Vision] No text extracted from {path.name}")
            return LoadedDocument(text="", source_file=path.name, source_type=source_type)

        # Normalize overly long table separators
        text = _normalize_table_separators(text)

        # Save to cache
        self._set_cache(path.name, img_data, text)

        print(f"  [Vision] Extracted {len(text)} chars from {path.name}")
        return LoadedDocument(
            text=text,
            source_file=path.name,
            source_type=source_type,
            metadata={"image_ocr": True},
        )

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
        import re
        import httpx

        raw = path.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        text = raw.decode(encoding)

        # Auto-extract text from image URLs in markdown
        image_urls = re.findall(
            r"!\[.*?\]\((https?://[^\s)]+\.(?:jpg|jpeg|png|gif|webp)[^\s)]*)\)",
            text, re.IGNORECASE,
        )
        if image_urls and "## 이미지에서 추출된 내용" not in text:
            extracted = []
            for url in image_urls:
                try:
                    img_data = httpx.get(url, timeout=15, follow_redirects=True).content
                    if len(img_data) < 1000:
                        continue

                    # Check cache
                    cached_text = self._get_cached_text(url, img_data)
                    if cached_text is not None:
                        extracted.append(cached_text)
                        continue

                    # Cache miss - call Gemini Vision
                    if self._genai_client is None:
                        from src.config import get_settings
                        settings = get_settings()
                        self._genai_client = genai.Client(
                            vertexai=True,
                            project=settings.gcp_project_id,
                            location=settings.llm_location or settings.gcp_location,
                        )
                        self._llm_model = settings.llm_model

                    print(f"  [Vision] Extracting text from URL: {url}")
                    response = self._genai_client.models.generate_content(
                        model=self._llm_model,
                        contents=Content(role="user", parts=[
                            Part(inline_data={"mime_type": "image/jpeg", "data": base64.b64encode(img_data).decode()}),
                            Part(text="""이 이미지에서 텍스트를 추출해 주세요. 다음 규칙을 따르세요:
1. 표(시간표, 프로그램 등)는 간결한 마크다운 표로 변환하세요. 구분선은 반드시 '| --- |' 형태로 짧게 유지하고, 절대 '-' 문자를 10개 이상 반복하지 마세요. 셀 내용에 과도한 공백(padding)을 넣지 마세요.
2. 일반 텍스트(제목, 장소, 안내문 등)는 그대로 텍스트로 출력하세요.
3. 표와 텍스트를 구분하여 정리하세요."""),
                        ]),
                        config=GenerateContentConfig(
                            max_output_tokens=8192,
                            thinking_config=ThinkingConfig(thinking_budget=0),
                        ),
                    )
                    if response.text:
                        normalized = _normalize_table_separators(response.text)
                        extracted.append(normalized)
                        self._set_cache(url, img_data, normalized)
                        print(f"  [Vision] Extracted {len(normalized)} chars")
                except Exception as e:
                    print(f"  [Vision] Error: {e}")

            if extracted:
                text += "\n\n## 이미지에서 추출된 내용\n\n" + "\n\n---\n\n".join(extracted)

        return LoadedDocument(
            text=text,
            source_file=path.name,
            source_type=source_type,
        )
