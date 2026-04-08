from src.chunker import DocumentChunker
from src.models import LoadedDocument


def test_chunk_medical_document(settings):
    chunker = DocumentChunker(settings)
    doc = LoadedDocument(
        text="진단: 고혈압 2기\n환자는 6개월 전부터 두통을 호소하였습니다.\n\n처방: 아물로디핀 5mg 1일 1회\n복용 시 주의사항을 안내하였습니다.",
        source_file="diagnosis.pdf",
        source_type="medical",
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    assert all(c.source_type == "medical" for c in chunks)
    assert all(c.source_file == "diagnosis.pdf" for c in chunks)


def test_chunk_event_document(settings):
    chunker = DocumentChunker(settings)
    doc = LoadedDocument(
        text="2024년 건강검진 행사 안내\n\n일시: 2024년 3월 15일\n장소: 본관 1층 대강당\n\n참가 신청은 홈페이지에서 가능합니다.",
        source_file="event_notice.txt",
        source_type="event",
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    assert all(c.source_type == "event" for c in chunks)


def test_medical_section_splitting(settings):
    chunker = DocumentChunker(settings)
    text = "진단: 당뇨병\n혈당 수치가 높습니다.\n\n처방: 메트포르민 500mg\n식후 복용하세요."

    sections = chunker._split_by_sections(text)

    assert len(sections) == 2
    assert sections[0][0] == "진단"
    assert sections[1][0] == "처방"


def test_empty_document(settings):
    chunker = DocumentChunker(settings)
    doc = LoadedDocument(
        text="",
        source_file="empty.txt",
        source_type="event",
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0
