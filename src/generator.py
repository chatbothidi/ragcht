from collections.abc import Generator

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from src.config import Settings

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
11. 마크다운 표의 빈 셀은 절대 쉼표(,)로 표현하지 마세요. 빈 셀이 있는 행은 해당 텍스트만 출력하고 나머지는 생략하세요. 예: "| **심포지엄 I** | | |" 같은 병합/빈 셀 행은 "심포지엄 I"로만 출력하세요."""

MAX_CONTINUATION_ROUNDS = 3


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
        return GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8192,
            system_instruction=SYSTEM_PROMPT,
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
            if page:
                header += f", 페이지 {page}"

            part = f"{header}\n{text}"

            # Include attachment info
            attachments = doc.get("attachments", [])
            if attachments:
                part += f"\n첨부파일: {', '.join(attachments)}"

            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)

    def generate(
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

        # First call
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        text = response.text or ""
        finish_reason = self._get_finish_reason(response)

        # If MAX_TOKENS, continue generating
        for _ in range(MAX_CONTINUATION_ROUNDS):
            if finish_reason != "MAX_TOKENS" or not text:
                break

            contents.append(Content(role="model", parts=[Part(text=text)]))
            contents.append(Content(role="user", parts=[Part(text="이어서 답변해 주세요.")]))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            continuation = response.text or ""
            finish_reason = self._get_finish_reason(response)

            if continuation:
                text += continuation

        return text

    def generate_stream(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
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

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=self._config(temperature),
        ):
            if chunk.text:
                yield chunk.text

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

    def rewrite_query(
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

        prompt = f"""이전 대화:
{history_text}

현재 질문: {query}

위 대화 맥락을 고려하여, 현재 질문을 독립적으로 이해할 수 있도록 다시 작성해 주세요.

규칙:
1. 대명사나 생략된 주어("그것", "거기서" 등)는 구체적으로 바꿔주세요.
2. 단, 현재 질문에 특정 연도, 날짜, 조건이 명시되지 않았다면 이전 대화의 시간/조건을 강제로 추가하지 마세요.
3. 현재 질문이 새로운 주제를 묻는 것이라면 이전 대화의 맥락을 적용하지 마세요.
4. 리라이팅된 질문만 출력하세요."""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=200,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
        result = (response.text or "").strip()
        if not result or len(result) < 5:
            return query
        return result
