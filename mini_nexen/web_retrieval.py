from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote_plus
from xml.etree import ElementTree

import math
import os
import requests


USER_AGENT = "mini-nexen/0.1 (+https://example.com)"
DEFAULT_TIMEOUT = 15
MAX_CONTENT_CHARS = 20000


@dataclass
class WebResult:
    title: str
    url: str
    text: str
    source: str


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) > MAX_CONTENT_CHARS:
        return cleaned[:MAX_CONTENT_CHARS] + "..."
    return cleaned


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return _clean_text(html.unescape(text))


def fetch_url_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        return _strip_html(resp.text)
    return _clean_text(resp.text)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_texts(
    texts: list[str],
    model: str,
    base_url: str,
    timeout: int,
) -> list[list[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("data", []):
        embedding = item.get("embedding")
        if isinstance(embedding, list):
            results.append([float(x) for x in embedding])
    return results


def _embed_texts_gemini(
    texts: list[str],
    model: str,
    api_key: str | None = None,
) -> list[list[float]]:
    try:
        from google import genai  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("google-genai is required for Gemini embeddings.") from exc

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    result = client.models.embed_content(model=model, contents=texts)
    embeddings = getattr(result, "embeddings", None) or []
    output: list[list[float]] = []
    for embedding in embeddings:
        values = None
        if isinstance(embedding, dict):
            values = embedding.get("values") or embedding.get("embedding")
        else:
            values = getattr(embedding, "values", None)
        if values is None and isinstance(embedding, list):
            values = embedding
        if values:
            output.append([float(x) for x in values])
    return output


def _resolve_embed_model(base_url: str, timeout: int) -> str | None:
    url = base_url.rstrip("/") + "/models"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "")
        lowered = model_id.lower()
        if "embed" in lowered or "embedding" in lowered:
            return model_id
    return None


def _semantic_rerank(
    query: str,
    results: list[WebResult],
    model: str,
    base_url: str,
    timeout: int,
) -> list[WebResult]:
    if not results:
        return results
    texts = [query]
    for result in results:
        text = result.text or ""
        if len(text) > 2000:
            text = text[:2000]
        texts.append(f"{result.title}\n{text}".strip())
    embeddings = _embed_texts(texts, model=model, base_url=base_url, timeout=timeout)
    if len(embeddings) != len(texts):
        return results
    query_vec = embeddings[0]
    scored = []
    for idx, result in enumerate(results, start=1):
        sim = _cosine_similarity(query_vec, embeddings[idx])
        scored.append((sim, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored]


def _semantic_rerank_gemini(
    query: str,
    results: list[WebResult],
    model: str,
    api_key: str | None,
) -> list[WebResult]:
    if not results:
        return results
    texts = [query]
    for result in results:
        text = result.text or ""
        if len(text) > 2000:
            text = text[:2000]
        texts.append(f"{result.title}\n{text}".strip())
    embeddings = _embed_texts_gemini(texts, model=model, api_key=api_key)
    if len(embeddings) != len(texts):
        return results
    query_vec = embeddings[0]
    scored = []
    for idx, result in enumerate(results, start=1):
        sim = _cosine_similarity(query_vec, embeddings[idx])
        scored.append((sim, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored]


def _dedupe_results(results: list[WebResult]) -> list[WebResult]:
    deduped: dict[str, WebResult] = {}
    for result in results:
        key = result.url.strip()
        if not key:
            continue
        current = deduped.get(key)
        if not current:
            deduped[key] = result
            continue
        if len(result.text or "") > len(current.text or ""):
            deduped[key] = result
    return list(deduped.values())


def expand_queries(
    query: str,
    modes: Iterable[str],
    max_queries: int,
    extra_queries: list[str] | None = None,
) -> list[str]:
    base = query.strip()
    if not base:
        return []
    if max_queries <= 1:
        return [base]

    modes_set = {mode.strip().lower() for mode in modes}
    tech_suffixes = [
        "technical report",
        "whitepaper",
        "release blog",
        "blog",
        "documentation",
        "forum",
    ]
    lit_suffixes = [
        "paper",
        "preprint",
        "arxiv",
        "conference",
        "journal",
        "survey",
    ]

    candidates: list[str] = []
    if "tech" in modes_set or "web" in modes_set:
        candidates.extend([f"{base} {suffix}" for suffix in tech_suffixes])
    if "lit" in modes_set or "literature" in modes_set:
        candidates.extend([f"{base} {suffix}" for suffix in lit_suffixes])

    queries = [base]
    seen = {base.casefold()}
    if extra_queries:
        for item in extra_queries:
            item = item.strip()
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            queries.append(item)
            seen.add(key)
            if len(queries) >= max_queries:
                return queries
    for item in candidates:
        key = item.casefold()
        if key in seen:
            continue
        queries.append(item)
        seen.add(key)
        if len(queries) >= max_queries:
            break
    return queries


def search_duckduckgo(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> list[WebResult]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    html_text = resp.text

    results: list[WebResult] = []
    seen = set()
    for match in re.finditer(r'class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_text):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))
        if not link.startswith("http"):
            continue
        if link in seen:
            continue
        seen.add(link)
        results.append(WebResult(title=title or link, url=link, text="", source="duckduckgo"))
        if len(results) >= max_results:
            break
    return results


def search_arxiv(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> list[WebResult]:
    q = quote_plus(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    feed = ElementTree.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[WebResult] = []
    for entry in feed.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        link = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        if not link:
            continue
        results.append(
            WebResult(
                title=_clean_text(title),
                url=link,
                text=_clean_text(summary),
                source="arxiv",
            )
        )
    return results


def search_semantic_scholar(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> list[WebResult]:
    q = quote_plus(query)
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={q}&limit={max_results}&fields=title,abstract,url,venue,year"
    )
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results: list[WebResult] = []
    for item in data.get("data", []):
        title = _clean_text(item.get("title") or "")
        abstract = _clean_text(item.get("abstract") or "")
        url = item.get("url") or ""
        if not url:
            continue
        results.append(WebResult(title=title or url, url=url, text=abstract, source="semantic_scholar"))
    return results


def search_crossref(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> list[WebResult]:
    q = quote_plus(query)
    url = f"https://api.crossref.org/works?query={q}&rows={max_results}"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("message", {}).get("items", [])
    results: list[WebResult] = []
    for item in items:
        title_list = item.get("title") or []
        title = _clean_text(title_list[0] if title_list else "")
        url = item.get("URL") or ""
        abstract = item.get("abstract") or ""
        abstract = _strip_html(abstract) if abstract else ""
        if not url:
            continue
        results.append(WebResult(title=title or url, url=url, text=_clean_text(abstract), source="crossref"))
    return results


def run_web_retrieval(
    query: str,
    modes: Iterable[str],
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
    fetch_pages: bool = True,
    hybrid: bool = False,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    embed_base_url: str | None = None,
    embed_timeout: int | None = None,
    embed_api_key: str | None = None,
    expand_query_flag: bool = True,
    max_queries: int = 4,
    extra_queries: list[str] | None = None,
) -> list[WebResult]:
    results: list[WebResult] = []
    modes_set = {mode.strip().lower() for mode in modes}

    queries = [query]
    if expand_query_flag:
        queries = expand_queries(query, modes_set, max_queries=max_queries, extra_queries=extra_queries)
    if not queries:
        return results

    for q in queries:
        if "tech" in modes_set or "web" in modes_set:
            results.extend(search_duckduckgo(q, max_results=max_results, timeout=timeout))

        if "lit" in modes_set or "literature" in modes_set:
            results.extend(search_arxiv(q, max_results=max_results, timeout=timeout))
            results.extend(search_semantic_scholar(q, max_results=max_results, timeout=timeout))
            results.extend(search_crossref(q, max_results=max_results, timeout=timeout))

    fetched: list[WebResult] = []
    for result in results:
        if not fetch_pages or result.text:
            fetched.append(result)
            continue
        try:
            text = fetch_url_text(result.url, timeout=timeout)
            fetched.append(WebResult(title=result.title, url=result.url, text=text, source=result.source))
            time.sleep(0.2)
        except Exception:
            fetched.append(result)
    merged = _dedupe_results(fetched)
    if not hybrid:
        return merged

    provider = (embed_provider or os.getenv("MINI_NEXEN_EMBED_PROVIDER") or "lmstudio").lower()
    model = embed_model or os.getenv("MINI_NEXEN_EMBED_MODEL")
    if provider == "gemini":
        if not model:
            model = "gemini-embedding-001"
        try:
            return _semantic_rerank_gemini(
                query=query,
                results=merged,
                model=model,
                api_key=embed_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            )
        except Exception:
            return merged

    base_url = embed_base_url or os.getenv("MINI_NEXEN_EMBED_BASE_URL") or os.getenv("LMSTUDIO_BASE_URL")
    if not base_url:
        return merged
    if not model:
        try:
            model = _resolve_embed_model(base_url, timeout=embed_timeout or timeout)
        except Exception:
            return merged
    if not model:
        return merged

    try:
        return _semantic_rerank(
            query=query,
            results=merged,
            model=model,
            base_url=base_url,
            timeout=embed_timeout or timeout,
        )
    except Exception:
        return merged
