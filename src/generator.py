import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator
from datetime import date

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from src.config import Settings

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    genai_errors.ServerError,
    httpx.TimeoutException,
    httpx.ConnectError,
)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        return True
    return False

SYSTEM_PROMPT = """당신은 의료 문서와 행사 문서를 기반으로 답변하는 전문 AI 어시스턴트입니다.

규칙:
1. 반드시 제공된 컨텍스트의 정보만을 사용하여 답변하세요.
2. 컨텍스트에 답변할 수 있는 정보가 없으면 어떤 맥락으로 검색했는지 설명하고, 관련 정보를 찾지 못했다고 답변하세요. 예: "이전 대화의 '대한임상보험의학회' 맥락에서 2023년 학술대회를 검색했으나, 관련 정보를 찾을 수 없습니다."
3. 출처 번호([1], [2] 등)를 본문에 표기하지 마세요.
4. 존댓말을 사용하세요.
5. 의료 정보는 정확하게 전달하세요.
6. 간결하고 명확하게 답변하세요.
7. 내용이 길 경우 핵심만 요약하여 답변하세요. 전체 내용을 나열하지 말고 중요한 포인트 위주로 정리하세요.
8. 컨텍스트에 여러 개의 관련 문서 조각이 있으면 모두 종합하여 답변하세요. 예를 들어 Room 1과 Room 2가 별도 조각에 있으면 둘 다 포함하세요.
9. 숫자 범위를 표시할 때 반드시 "30~40"처럼 ~ 기호를 포함하세요. "3040"처럼 붙여 쓰지 마세요.
10. 컨텍스트에 첨부파일(신청서, 양식 등)이 언급되면 다운로드 링크를 제공하세요. 반드시 다음 형식만 사용하세요: [[다운로드:파일명.확장자]]
    - 마크다운 링크 문법 [텍스트](URL) 은 사용하지 마세요.
    - 파일명에 괄호, 공백, 한글이 포함되어도 원본 그대로 적으세요. URL 인코딩하지 마세요.
    - 예: [[다운로드:결핵진료지침(4판)_(Web용)_최종.pdf]]
11. 마크다운 표의 빈 셀은 절대 쉼표(,)로 표현하지 마세요. 빈 셀이 있는 행은 해당 텍스트만 출력하고 나머지는 생략하세요. 예: "| **심포지엄 I** | | |" 같은 병합/빈 셀 행은 "심포지엄 I"로만 출력하세요.
12. 사용자가 "최근", "최신", "요즘" 등 시점을 묻는 경우, 컨텍스트의 [연도: YYYY] 값을 기준으로 연도가 큰 문서(최신)부터 나열하세요. 연도 정보가 없는 문서는 뒤에 배치하세요.
13. 사용자가 "정렬", "내림차순", "오름차순", "날짜순", "일자순", "나열", "리스트" 등 정렬/목록화를 요구하면 다음을 반드시 지키세요.
    - 컨텍스트에 등장하는 순서를 무시하고 지정된 기준(날짜, 연도 등)으로 **모든 항목을 전부 정렬**하세요.
    - 앞쪽만 정렬하고 뒤쪽을 그대로 이어붙이는 실수를 하지 마세요.
    - 같은 행사(같은 게시글)는 한 번만 나열하고, 날짜/장소/주요 내용 정도로 간결히 요약하세요.
    - 정렬 기준이 날짜인데 일/월까지 명시된 경우 일 단위까지 비교하세요. 연도만 있으면 연도로 비교하세요.
14. 컨텍스트의 청크 헤더에 `[원문: https://...]`이 포함되어 있으면, 답변 끝에 한 줄로 `자세한 내용은 [원문 보기](URL)에서 확인하실 수 있습니다.` 형식으로 자연스럽게 안내하세요. 같은 게시글의 청크가 여러 개여도 동일 URL은 한 번만 표시하세요. URL이 없으면 이 안내를 생략하세요."""

MAX_CONTINUATION_ROUNDS = 3


async def _with_retry(
    coro_factory,
    *,
    label: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
):
    """Run an async callable with exponential backoff on retryable errors.

    Treats 429 (quota exhausted) as retryable with a longer baseline delay.
    """
    delay = initial_delay
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt == max_attempts:
                break
            is_quota = (
                isinstance(exc, genai_errors.ClientError)
                and getattr(exc, "code", None) == 429
            )
            base = max(delay, 8.0) if is_quota else delay
            sleep_for = base + random.uniform(0, base * 0.25)
            logger.warning(
                "[%s] %s on attempt %d/%d; retrying in %.1fs",
                label,
                type(exc).__name__,
                attempt,
                max_attempts,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)
            delay *= 2
    raise last_exc  # type: ignore[misc]


class LLMGenerator:
    def __init__(self, settings: Settings):
        llm_location = settings.llm_location or settings.gcp_location
        self.client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=llm_location,
        )
        self.model_name = settings.llm_model
        self.max_context_tokens = settings.max_context_tokens

    def _config(self, temperature: float = 0.3) -> GenerateContentConfig:
        today = date.today().isoformat()
        system_instruction = f"오늘 날짜: {today}\n\n{SYSTEM_PROMPT}"
        return GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8192,
            system_instruction=system_instruction,
            thinking_config=ThinkingConfig(thinking_budget=0),
        )

    def build_context(self, documents: list[dict]) -> str:
        """Build context string from retrieved documents with source markers."""
        context_parts = []

        for i, doc in enumerate(documents, start=1):
            source = doc.get("source_file", "Unknown")
            page = doc.get("page_number")
            text = doc.get("text", "")

            header = f"[{i}] 출처: {source}"
            if doc.get("post_title"):
                header += f" (게시글: {doc['post_title']})"
            if doc.get("year") is not None:
                header += f" [연도: {doc['year']}]"
            if page:
                header += f", 페이지 {page}"
            if doc.get("url"):
                header += f" [원문: {doc['url']}]"

            part = f"{header}\n{text}"

            attachments = doc.get("attachments", [])
            if attachments:
                part += f"\n첨부파일: {', '.join(attachments)}"

            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)

    async def generate(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate response with automatic continuation on MAX_TOKENS."""
        contents = []

        if conversation_history:
            for turn in conversation_history[-6:]:
                contents.append(Content(role=turn["role"], parts=[Part(text=turn["content"])]))

        user_message = f"""컨텍스트:
{context}

질문: {query}

위 컨텍스트를 기반으로 답변해 주세요."""

        contents.append(Content(role="user", parts=[Part(text=user_message)]))

        config = self._config(temperature)

        response = await _with_retry(
            lambda: self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            ),
            label="generate",
        )
        text = response.text or ""
        finish_reason = self._get_finish_reason(response)

        for _ in range(MAX_CONTINUATION_ROUNDS):
            if finish_reason != "MAX_TOKENS" or not text:
                break

            contents.append(Content(role="model", parts=[Part(text=text)]))
            contents.append(Content(role="user", parts=[Part(text="이어서 답변해 주세요.")]))

            response = await _with_retry(
                lambda: self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                ),
                label="generate-continue",
            )
            continuation = response.text or ""
            finish_reason = self._get_finish_reason(response)

            if continuation:
                text += continuation

        return text

    async def generate_stream(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        contents = []

        if conversation_history:
            for turn in conversation_history[-6:]:
                contents.append(Content(role=turn["role"], parts=[Part(text=turn["content"])]))

        user_message = f"""컨텍스트:
{context}

질문: {query}

위 컨텍스트를 기반으로 답변해 주세요."""

        contents.append(Content(role="user", parts=[Part(text=user_message)]))

        stream = await _with_retry(
            lambda: self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=self._config(temperature),
            ),
            label="generate-stream",
        )

        start = time.monotonic()
        first_chunk_at: float | None = None
        chunk_count = 0
        empty_chunk_count = 0
        yielded_chars = 0
        finish_reason: str | None = None
        block_reason: str | None = None

        async for chunk in stream:
            chunk_count += 1
            if first_chunk_at is None:
                first_chunk_at = time.monotonic() - start

            try:
                if chunk.candidates:
                    fr = chunk.candidates[0].finish_reason
                    if fr is not None:
                        finish_reason = fr.name if hasattr(fr, "name") else str(fr)
            except (AttributeError, IndexError):
                pass

            try:
                pf = getattr(chunk, "prompt_feedback", None)
                if pf and getattr(pf, "block_reason", None):
                    br = pf.block_reason
                    block_reason = br.name if hasattr(br, "name") else str(br)
            except AttributeError:
                pass

            if chunk.text:
                yielded_chars += len(chunk.text)
                yield chunk.text
            else:
                empty_chunk_count += 1

        total_elapsed = time.monotonic() - start
        first = first_chunk_at if first_chunk_at is not None else -1.0
        if yielded_chars == 0:
            logger.warning(
                "[generate-stream] EMPTY response | chunks=%d empty=%d "
                "first_chunk=%.2fs total=%.2fs finish_reason=%s block_reason=%s "
                "ctx_len=%d query=%r",
                chunk_count, empty_chunk_count, first, total_elapsed,
                finish_reason, block_reason, len(context), query[:80],
            )
        else:
            logger.info(
                "[generate-stream] OK | chunks=%d chars=%d "
                "first_chunk=%.2fs total=%.2fs finish_reason=%s",
                chunk_count, yielded_chars, first, total_elapsed, finish_reason,
            )

    @staticmethod
    def _get_finish_reason(response) -> str:
        """Get finish reason from response."""
        try:
            if response.candidates:
                reason = response.candidates[0].finish_reason
                if reason and reason.name == "MAX_TOKENS":
                    return "MAX_TOKENS"
                return "STOP"
        except (AttributeError, IndexError):
            pass
        return "UNKNOWN"

    async def rewrite_query(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """Rewrite query using conversation context to resolve references."""
        if not conversation_history:
            return query

        history_text = "\n".join(
            f"{'사용자' if t['role'] == 'user' else 'AI'}: {t['content']}"
            for t in conversation_history[-4:]
        )

        today_iso = date.today().isoformat()
        current_year = date.today().year

        prompt = f"""오늘 날짜: {today_iso} (현재 연도: {current_year}년)

이전 대화:
{history_text}

현재 질문: {query}

위 대화 맥락을 고려하여, 현재 질문을 독립적으로 이해할 수 있도록 다시 작성해 주세요.

규칙:
1. 대명사나 생략된 주어("그것", "거기서" 등)는 구체적으로 바꿔주세요.
2. "올해", "작년", "내년", "최근", "요즘" 같은 상대 시간 표현은 위에 명시된 오늘 날짜를 기준으로 절대 연도(예: "{current_year}년")로 변환하세요. 이전 대화에 등장한 연도가 아니라 반드시 오늘 날짜 기준으로 계산하세요.
3. 단, 현재 질문에 특정 연도, 날짜, 조건이 명시되지 않았고 상대 시간 표현도 없다면 이전 대화의 시간/조건을 강제로 추가하지 마세요.
4. 현재 질문이 새로운 주제를 묻는 것이라면 이전 대화의 맥락을 적용하지 마세요.
5. 리라이팅된 질문만 출력하세요."""

        response = await _with_retry(
            lambda: self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=200,
                    thinking_config=ThinkingConfig(thinking_budget=0),
                ),
            ),
            label="rewrite-query",
        )
        result = (response.text or "").strip()
        if not result or len(result) < 5:
            return query
        return result
