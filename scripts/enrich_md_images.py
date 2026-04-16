#!/usr/bin/env python3
"""Enrich markdown files by extracting text from image URLs using Gemini Vision.

Finds image URLs in markdown files (![...](url) or [![...](url)](...)),
extracts text from each image, and appends the extracted text to the file.

Usage:
    python scripts/enrich_md_images.py <md_file>
    python scripts/enrich_md_images.py data/documents/events/some_file.md
"""

import argparse
import base64
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from src.config import get_settings

# Match markdown image patterns: ![alt](url) or [![alt](url)](link)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\((https?://[^\s)]+\.(?:jpg|jpeg|png|gif|webp)[^\s)]*)\)", re.IGNORECASE)


def extract_text_from_image(client, model_name, url):
    """Download image and extract text using Gemini Vision."""
    try:
        img_data = httpx.get(url, timeout=15, follow_redirects=True).content
        if len(img_data) < 1000:
            print(f"  Skipped (too small: {len(img_data)} bytes)")
            return None

        response = client.models.generate_content(
            model=model_name,
            contents=Content(
                role="user",
                parts=[
                    Part(inline_data={"mime_type": "image/jpeg", "data": base64.b64encode(img_data).decode()}),
                    Part(text="""이 이미지에서 텍스트를 추출해 주세요. 다음 규칙을 따르세요:
1. 표(시간표, 프로그램 등)가 있으면 마크다운 표(| 시간 | 내용 | 형식)로 변환하세요.
2. 일반 텍스트(제목, 장소, 안내문 등)는 그대로 텍스트로 출력하세요.
3. 표와 텍스트를 구분하여 정리하세요."""),
                ],
            ),
            config=GenerateContentConfig(
                max_output_tokens=8192,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text or None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Enrich markdown files with image text")
    parser.add_argument("file", help="Path to markdown file")
    args = parser.parse_args()

    md_path = Path(args.file)
    if not md_path.exists():
        print(f"File not found: {md_path}")
        sys.exit(1)

    content = md_path.read_text(encoding="utf-8")

    # Find all image URLs
    image_urls = IMAGE_PATTERN.findall(content)
    if not image_urls:
        print("No image URLs found in file.")
        return

    print(f"Found {len(image_urls)} image(s) in {md_path.name}")

    settings = get_settings()
    client = genai.Client(
        vertexai=True,
        project=settings.gcp_project_id,
        location=settings.llm_location or settings.gcp_location,
    )

    extracted_texts = []
    for i, url in enumerate(image_urls, 1):
        print(f"\n[{i}/{len(image_urls)}] {url}")
        text = extract_text_from_image(client, settings.llm_model, url)
        if text:
            extracted_texts.append(text)
            print(f"  Extracted {len(text)} chars")
        else:
            print(f"  No text extracted")

    if not extracted_texts:
        print("\nNo text was extracted from any image.")
        return

    # Append extracted text to the file
    addition = "\n\n## 이미지에서 추출된 내용\n\n"
    addition += "\n\n---\n\n".join(extracted_texts)

    new_content = content + addition
    md_path.write_text(new_content, encoding="utf-8")

    print(f"\nDone! Appended extracted text to {md_path.name}")
    print(f"  Images processed: {len(image_urls)}")
    print(f"  Text extracted: {len(extracted_texts)}")


if __name__ == "__main__":
    main()
