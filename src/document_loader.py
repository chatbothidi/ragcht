import base64
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import re
import chardet
import fitz  # pymupdf
fitz.TOOLS.mupdf_display_errors(False)
from docx import Document as DocxDocument
from google.cloud import documentai_v1 as documentai
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from src.config import get_settings
from src.models import LoadedDocument

# markdown 변환 시 문제되는 특수 기호들
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
    """pdf의 특수기호들 markdown에 맞게 변환."""
    return text.translate(_NORMALIZE_MAP)

def _normalize_table_separators(text: str) -> str:
    """긴 내용의 마크다운 테이블 짧게 변환

    '|:----...----:|' (300자 이상) 형태의 패턴을 '|:---:|' 형태로 변환.
    """
    return re.sub(r":?-{10,}:?", lambda m: ":---:" if ":" in m.group() else "---", text)

class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    CACHE_FILENAME = ".image_cache.json"

    def __init__(self, genai_client=None):
        self._docai_client = None
        self._docai_processor: str | None = None
        self._genai_client = genai_client
        self._llm_model: str | None = None
        self._current_cache: dict = {}
        self._current_cache_path: Path | None = None
        self._cache_dirty = False

    def _ensure_genai_client(self) -> None:
        """클라이언트가 주입되지 않은 경우 settings에서 Vertex genai 클라이언트를 lazy하게 생성."""
        if self._genai_client is not None:
            return
        settings = get_settings()
        self._genai_client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.llm_location or settings.gcp_location,
        )
        self._llm_model = settings.llm_model

    def _ensure_docai_client(self) -> None:
        """settings에서 Document AI 클라이언트와 프로세서 이름을 lazy하게 로드."""
        if self._docai_client is not None:
            return
        settings = get_settings()
        self._docai_client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": "us-documentai.googleapis.com"}
        )
        self._docai_processor = settings.docai_processor

    def _vision_generate_with_retry(self, contents, *, label: str, max_attempts: int = 3):
        """Sync retry wrapper for Gemini Vision generate_content.

        Retries 429 (RESOURCE_EXHAUSTED) and 5xx server errors with backoff.
        Raises on non-retryable errors or after max_attempts.
        """
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._genai_client.models.generate_content(
                    model=self._llm_model,
                    contents=contents,
                    config=GenerateContentConfig(
                        max_output_tokens=8192,
                        thinking_config=ThinkingConfig(thinking_budget=0),
                    ),
                )
            except genai_errors.ClientError as exc:
                if getattr(exc, "code", None) != 429:
                    raise
                last_exc = exc
                if attempt == max_attempts:
                    break
                sleep_for = max(delay, 8.0) + random.uniform(0, 2.0)
                print(f"  [Vision] 429 on {label} attempt {attempt}/{max_attempts}; retrying in {sleep_for:.1f}s")
                time.sleep(sleep_for)
                delay *= 2
            except genai_errors.ServerError as exc:
                last_exc = exc
                if attempt == max_attempts:
                    break
                sleep_for = delay + random.uniform(0, delay * 0.25)
                print(f"  [Vision] ServerError on {label} attempt {attempt}/{max_attempts}; retrying in {sleep_for:.1f}s")
                time.sleep(sleep_for)
                delay *= 2
        raise last_exc  # type: ignore[misc]

    def _load_image_cache(self, post_dir: Path) -> dict:
        """게시물 디렉터리의 이미지 캐시 로드."""
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
        """변경된 이미지 캐시 저장."""
        if self._cache_dirty and self._current_cache_path:
            with open(self._current_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._current_cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """바이너리 데이터의 MD5 해시 계산."""
        return hashlib.md5(data).hexdigest()

    def _get_cached_text(self, key: str, data: bytes) -> str | None:
        """캐시에서 추출된 텍스트 확인. 캐시된 텍스트 또는 None 반환."""
        file_hash = self._compute_hash(data)
        cached = self._current_cache.get(key)
        if cached and cached.get("hash") == file_hash:
            print(f"  [Cache hit] {key}")
            return cached["extracted_text"]
        return None

    def _set_cache(self, key: str, data: bytes, text: str) -> None:
        """추출된 텍스트를 캐시에 저장."""
        self._current_cache[key] = {
            "hash": self._compute_hash(data),
            "extracted_text": text,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cache_dirty = True

    def load_file(self, file_path: str, source_type: str, post_id: int | None = None,
                  post_title: str | None = None, attachments: list[str] | None = None,
                  year: int | None = None, url: str | None = None) -> LoadedDocument:
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

        # 게시물 메타데이터 추가
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
        if url:
            doc.metadata["url"] = url

        return doc

    def load_directory(self, directory: str) -> list[LoadedDocument]:
        """게시물 ID 기반 디렉터리 구조에서 문서 로드."""
        docs = []
        base_path = Path(directory)

        for post_dir in sorted(base_path.iterdir()):
            if not post_dir.is_dir():
                continue

            # 메타데이터를 위해 data.json 읽기
            data_json = post_dir / "data.json"
            if not data_json.exists():
                # Fallback: 이전 구조 (medical/events 폴더)
                if post_dir.name in ("medical", "events"):
                    docs.extend(self._load_old_structure(post_dir))
                continue

            with open(data_json, encoding="utf-8") as f:
                metadata = json.load(f)

            # 해당 게시물의 이미지 캐시 로드
            self._load_image_cache(post_dir)

            post_id = metadata.get("id")
            category = metadata.get("category", "unknown")
            title = metadata.get("title", "")
            main_file = metadata.get("main_file", "")
            attachments = metadata.get("attachments", [])
            year = metadata.get("year")
            url = metadata.get("url")

            all_supported = self.SUPPORTED_EXTENSIONS | self.IMAGE_EXTENSIONS

            # 메인 파일 로드
            main_path = post_dir / main_file
            if main_path.exists() and main_path.suffix.lower() in all_supported:
                doc = self.load_file(
                    str(main_path), category,
                    post_id=post_id, post_title=title,
                    attachments=attachments, year=year, url=url,
                )
                docs.append(doc)

            # 첨부 파일 로드
            for att_name in attachments:
                att_path = post_dir / att_name
                if att_path.exists() and att_path.suffix.lower() in all_supported:
                    doc = self.load_file(
                        str(att_path), category,
                        post_id=post_id, post_title=title, year=year, url=url,
                    )
                    docs.append(doc)

            # 해당 게시물의 이미지 캐시 저장
            self._save_image_cache()

        return docs

    def _load_old_structure(self, subdir_path: Path) -> list[LoadedDocument]:
        """Fallback: 이전 medical/events 폴더 구조에서 로드."""
        docs = []
        source_type = "medical" if subdir_path.name == "medical" else "event"
        for file_path in subdir_path.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self.load_file(str(file_path), source_type)
                docs.append(doc)
        return docs

    @staticmethod
    def _is_low_quality_text(text: str) -> bool:
        """PDF에서 추출된 텍스트의 품질 검사."""
        if not text.strip():
            return True

        chars = len(text.replace(" ", "").replace("\n", ""))
        total = len(text)

        if chars == 0:
            return True

        # 공백 비율이 50% 초과 (불량)
        if 1 - (chars / total) > 0.5:
            return True

        # 단어 평균 길이가 2 미만 (불량)
        words = text.split()
        if words and sum(len(w) for w in words) / len(words) < 2:
            return True

        return False

    def _load_pdf(self, path: Path, source_type: str) -> LoadedDocument:
        pdf_data = path.read_bytes()

        # 캐시 확인
        cached_text = self._get_cached_text(path.name, pdf_data)
        if cached_text is not None:
            return LoadedDocument(
                text=cached_text,
                source_file=path.name,
                source_type=source_type,
                metadata={"ocr": True, "cached": True},
            )

        # 캐시 미스 - Document AI OCR 실행
        self._ensure_docai_client()

        # 페이지 수 확인 - Document AI online 제한은 30페이지
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

        # 캐시에 저장
        self._set_cache(path.name, pdf_data, full_text)

        return LoadedDocument(
            text=full_text,
            source_file=path.name,
            source_type=source_type,
            pages=pages,
            metadata={"total_pages": len(pages), "ocr": True},
        )

    def _ocr_pdf_bytes(self, pdf_data: bytes) -> tuple[str, list[dict]]:
        """30페이지 제한 내의 PDF OCR 처리."""
        raw_document = documentai.RawDocument(
            content=pdf_data,
            mime_type="application/pdf",
        )
        request = documentai.ProcessRequest(
            name=self._docai_processor,
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
        """큰 PDF를 페이지 + 사이즈 제한에 맞게 분할해 OCR 처리.

        Document AI 한도: 페이지 ≤15, raw 사이즈 ≤40MB. 페이지 분할만으론 사이즈 초과
        발생 가능(이미지 많은 PDF). 사이즈 초과 시 재귀적으로 절반 분할.
        """
        DOCAI_MAX_BYTES = 40 * 1024 * 1024  # 41943040 한도에 안전 마진
        all_text_parts: list[str] = []
        all_pages: list[dict] = []

        doc = fitz.open(str(path))

        def _extract_chunk_bytes(start: int, end: int) -> bytes:
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
            chunk_bytes = chunk_doc.tobytes()
            chunk_doc.close()
            return chunk_bytes

        def _process_range(start: int, end: int) -> None:
            chunk_bytes = _extract_chunk_bytes(start, end)
            size_mb = len(chunk_bytes) // (1024 * 1024)

            # 사이즈 초과 + 분할 가능하면 재귀
            if len(chunk_bytes) > DOCAI_MAX_BYTES and (end - start) > 1:
                mid = (start + end) // 2
                print(f"    Pages {start + 1}-{end}: {size_mb}MB exceeds limit, splitting at {mid + 1}")
                _process_range(start, mid)
                _process_range(mid, end)
                return

            print(f"    Pages {start + 1}-{end} ({size_mb}MB)...")
            text, pages = self._ocr_pdf_bytes(chunk_bytes)
            all_text_parts.append(text)
            for page in pages:
                page["page"] += start
            all_pages.extend(pages)

        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            _process_range(start, end)

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
        """Gemini Vision으로 이미지 파일에서 텍스트 추출 (캐싱 포함)."""
        img_data = path.read_bytes()

        # 캐시 확인
        cached_text = self._get_cached_text(path.name, img_data)
        if cached_text is not None:
            return LoadedDocument(
                text=cached_text, source_file=path.name, source_type=source_type,
                metadata={"image_ocr": True, "cached": True},
            )

        self._ensure_genai_client()

        print(f"  [Vision] Extracting text from image: {path.name}")

        ext = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
        mime_type = mime_map.get(ext, "image/jpeg")

        contents = Content(
            role="user",
            parts=[
                Part(inline_data={"mime_type": mime_type, "data": base64.b64encode(img_data).decode()}),
                Part(text="""이 이미지에서 텍스트를 추출해 주세요. 다음 규칙을 따르세요:
1. 표(시간표, 프로그램 등)는 간결한 마크다운 표로 변환하세요. 구분선은 반드시 '| --- |' 형태로 짧게 유지하고, 절대 '-' 문자를 10개 이상 반복하지 마세요. 셀 내용에 과도한 공백(padding)을 넣지 마세요.
2. 일반 텍스트(제목, 장소, 안내문 등)는 그대로 텍스트로 출력하세요.
3. 표와 텍스트를 구분하여 정리하세요."""),
            ],
        )
        try:
            response = self._vision_generate_with_retry(contents, label=f"image:{path.name}")
        except Exception as e:
            print(f"  [Vision] Failed after retries for {path.name}: {e}")
            return LoadedDocument(text="", source_file=path.name, source_type=source_type)

        text = response.text or ""
        if not text:
            print(f"  [Vision] No text extracted from {path.name}")
            return LoadedDocument(text="", source_file=path.name, source_type=source_type)

        # 너무 긴 테이블 구분선 정규화
        text = _normalize_table_separators(text)

        # 캐시에 저장
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

        # 마크다운 내 이미지 URL에서 텍스트 자동 추출
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

                    # 캐시 확인
                    cached_text = self._get_cached_text(url, img_data)
                    if cached_text is not None:
                        extracted.append(cached_text)
                        continue

                    # 캐시 미스 - Gemini Vision 호출
                    self._ensure_genai_client()

                    print(f"  [Vision] Extracting text from URL: {url}")
                    contents = Content(role="user", parts=[
                        Part(inline_data={"mime_type": "image/jpeg", "data": base64.b64encode(img_data).decode()}),
                        Part(text="""이 이미지에서 텍스트를 추출해 주세요. 다음 규칙을 따르세요:
1. 표(시간표, 프로그램 등)는 간결한 마크다운 표로 변환하세요. 구분선은 반드시 '| --- |' 형태로 짧게 유지하고, 절대 '-' 문자를 10개 이상 반복하지 마세요. 셀 내용에 과도한 공백(padding)을 넣지 마세요.
2. 일반 텍스트(제목, 장소, 안내문 등)는 그대로 텍스트로 출력하세요.
3. 표와 텍스트를 구분하여 정리하세요."""),
                    ])
                    response = self._vision_generate_with_retry(contents, label=f"url:{url[-40:]}")
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