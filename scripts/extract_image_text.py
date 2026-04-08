#!/usr/bin/env python3
"""Extract text from an image URL using Gemini Vision.

Usage:
    python scripts/extract_image_text.py <image_url>
    python scripts/extract_image_text.py <image_url> --output result.txt
"""

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

from src.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Extract text from image URL")
    parser.add_argument("url", help="Image URL to extract text from")
    parser.add_argument("--output", "-o", help="Save result to file")
    args = parser.parse_args()

    settings = get_settings()
    client = genai.Client(
        vertexai=True,
        project=settings.gcp_project_id,
        location=settings.llm_location or settings.gcp_location,
    )

    print(f"Downloading: {args.url}")
    img_data = httpx.get(args.url, timeout=15, follow_redirects=True).content
    print(f"Downloaded: {len(img_data)} bytes")

    print("Extracting text...")
    response = client.models.generate_content(
        model=settings.llm_model,
        contents=Content(
            role="user",
            parts=[
                Part(inline_data={"mime_type": "image/jpeg", "data": base64.b64encode(img_data).decode()}),
                Part(text="이 이미지에 있는 모든 텍스트를 그대로 추출해 주세요. 레이아웃을 최대한 유지하면서 텍스트만 출력하세요."),
            ],
        ),
        config=GenerateContentConfig(
            max_output_tokens=8192,
            thinking_config=ThinkingConfig(thinking_budget=0),
        ),
    )

    result = response.text or ""
    print()
    print(result)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
