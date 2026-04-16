#!/usr/bin/env python3
"""Migrate from medical/events folder structure to post ID-based structure."""

import json
import shutil
from pathlib import Path

DOCS_DIR = Path("data/documents")

# Define post groups: post_id -> {title, category, year, files (relative to current structure)}
POSTS = [
    {
        "id": 1,
        "title": "급성호흡곤란증후군(ARDS) 임상진료지침",
        "category": "medical",
        "year": 2016,
        "main_file": "medical/급성호흡곤란증후군(ARDS) 임상진료지침.md",
        "attachments": ["medical/ards_2016.pdf"],
    },
    {
        "id": 2,
        "title": "결핵진료지침 5판",
        "category": "medical",
        "year": 2024,
        "main_file": "medical/결핵진료지침 5판.md",
        "attachments": ["medical/결핵진료지침(5판)_내지 최종(전자용)_수정 240207 기준.pdf"],
    },
    {
        "id": 3,
        "title": "대한임상보험의학회 학술대회 개최",
        "category": "event",
        "year": 2008,
        "main_file": "events/대한임상보험의학회 학술대회 개최.md",
        "attachments": [
            "events/1209713385_참가신청서.docx",
            "events/1209713385_초청장.pdf",
        ],
    },
    {
        "id": 4,
        "title": "2023 대한폐암학회 춘계학술대회 개최 안내",
        "category": "event",
        "year": 2023,
        "main_file": "events/2023 대한폐암학회 춘계학술대회 개최 안내.md",
        "attachments": [],
    },
    {
        "id": 5,
        "title": "AIRWAY VISTA 2023 개최 안내",
        "category": "event",
        "year": 2023,
        "main_file": "events/AIRWAY VISTA 2023 개최 안내.md",
        "attachments": [],
    },
    {
        "id": 6,
        "title": "제14회 희귀질환·호흡재활 심포지엄 개최 안내",
        "category": "event",
        "year": 2021,
        "main_file": "events/제14회 희귀질환·호흡재활 심포지엄 개최 안내.md",
        "attachments": ["events/사전등록신청서 2021(대호연 2021-268).docx"],
    },
    {
        "id": 7,
        "title": "세계 패혈증의 날 심포지엄",
        "category": "event",
        "year": 2021,
        "main_file": "events/세계 패혈증의 날 심포지엄 - 질병관리청 패혈증 등록 사업 결과보고회_패혈증의 실태와 대책.md",
        "attachments": [],
    },
    {
        "id": 8,
        "title": "2020 동계 분자폐암연구회 임상연구 워크숍 개최 안내",
        "category": "event",
        "year": 2020,
        "main_file": "events/2020 동계 분자폐암연구회 임상연구 워크숍 개최 안내.md",
        "attachments": [],
    },
    {
        "id": 9,
        "title": "2020년 3월 28일 제4회 심포지엄",
        "category": "event",
        "year": 2020,
        "main_file": "events/2020년 3월 28일 제4회 심포지엄.md",
        "attachments": [],
    },
    {
        "id": 10,
        "title": "제16차 천식연구회·COPD연구회 공동심포지엄 개최 안내",
        "category": "event",
        "year": None,
        "main_file": "events/제16차 천식연구회 · COPD연구회 공동심포지엄 개최 안내.md",
        "attachments": [],
    },
    {
        "id": 11,
        "title": "제271회 대한결핵 및 호흡기학회 심포지엄 개최 안내",
        "category": "event",
        "year": None,
        "main_file": "events/제271회 대한결핵 및 호흡기학회 심포지엄 개최 안내.md",
        "attachments": [],
    },
    {
        "id": 12,
        "title": "천식 진료지침 공청회 개최 안내",
        "category": "event",
        "year": None,
        "main_file": "events/천식 진료지침 공청회 개최 안내.md",
        "attachments": [],
    },
]


def main():
    # Create new structure
    for post in POSTS:
        post_dir = DOCS_DIR / str(post["id"])
        post_dir.mkdir(parents=True, exist_ok=True)

        # Move main file
        src = DOCS_DIR / post["main_file"]
        if src.exists():
            dst = post_dir / src.name
            shutil.move(str(src), str(dst))
            print(f"  Moved: {src.name} -> {post['id']}/")

        # Move attachments
        attachment_names = []
        for att_path in post["attachments"]:
            src = DOCS_DIR / att_path
            if src.exists():
                dst = post_dir / src.name
                shutil.move(str(src), str(dst))
                attachment_names.append(src.name)
                print(f"  Moved: {src.name} -> {post['id']}/")

        # Create data.json
        data = {
            "id": post["id"],
            "title": post["title"],
            "category": post["category"],
            "year": post["year"],
            "main_file": Path(post["main_file"]).name,
            "attachments": attachment_names,
        }
        with open(post_dir / "data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Created: {post['id']}/data.json")

    # Remove old directories if empty
    for subdir in ["medical", "events"]:
        d = DOCS_DIR / subdir
        if d.exists() and not list(d.iterdir()):
            d.rmdir()
            print(f"  Removed empty: {subdir}/")

    print(f"\nMigration complete! {len(POSTS)} posts created.")


if __name__ == "__main__":
    main()
