from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")

ZH_EN_GLOSSARY = {
    "毕业": "graduation graduate credits diploma",
    "学分": "credits credit",
    "休学": "leave of absence loa",
    "请假": "leave of absence",
    "退课": "withdraw withdrawal drop add",
    "转专业": "change major major declaration",
    "专业": "major",
    "量化推理": "quantitative reasoning qr",
    "分布要求": "distribution requirement",
    "院长名单": "deans list dean's list honors",
    "留学": "study abroad global education",
    "补考": "make up retake",
    "不完整成绩": "incomplete grade",
    "课程重复": "course repeat repeat a course",
    "学业警告": "academic warning",
    "学业 probation": "academic probation",
    "停学": "academic suspension",
    "别的学校": "different universities transfer credit",
    "别的大学": "different universities transfer credit",
    "上课": "take classes courses enroll",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def expand_query(text: str) -> str:
    expanded = [text]
    lowered = text.lower()
    for key, value in ZH_EN_GLOSSARY.items():
        if key in lowered:
            expanded.append(value)
    return " ".join(expanded)


def hashed_embedding(text: str, dims: int = 512) -> list[float]:
    counts = Counter(tokenize(text))
    vector = [0.0] * dims
    for token, weight in counts.items():
        vector[hash(token) % dims] += float(weight)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
