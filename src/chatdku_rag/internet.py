from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request


def search_duckduckgo(query: str, limit: int = 3) -> list[dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<href>.*?)">(?P<title>.*?)</a>',
        re.IGNORECASE,
    )
    results = []
    for match in pattern.finditer(body):
        title = re.sub(r"<.*?>", "", html.unescape(match.group("title"))).strip()
        href = html.unescape(match.group("href")).strip()
        if title and href:
            results.append({"title": title, "url": href})
        if len(results) >= limit:
            break
    return results
