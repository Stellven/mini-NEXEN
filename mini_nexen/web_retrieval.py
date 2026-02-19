from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests

from .embeddings import EmbeddingClient, EmbeddingConfig, cosine_similarity
from .llm import log_task_event

USER_AGENT = "mini-nexen/0.1 (+https://example.com)"
DEFAULT_TIMEOUT = 15
MAX_CONTENT_CHARS = 20000
RETRY_MAX_SECONDS = 180
RETRY_BASE_SLEEP = 2
RETRY_MAX_SLEEP = 30
_REDDIT_TOKEN: str | None = None
_REDDIT_TOKEN_EXPIRES_AT = 0.0


class RetrievalRateLimitError(RuntimeError):
    def __init__(self, label: str, attempts: int, elapsed: float):
        self.label = label
        self.attempts = attempts
        self.elapsed = elapsed
        super().__init__(
            f"Rate limit from {label} after {attempts} attempts over {elapsed:.0f}s"
        )


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


def _request_with_retries(
    method: str,
    url: str,
    timeout: int,
    headers: dict[str, str] | None = None,
    data: str | None = None,
    auth: tuple[str, str] | None = None,
    label: str = "request",
    max_retry_seconds: int = RETRY_MAX_SECONDS,
) -> requests.Response:
    from .llm import log_task_event

    start = time.monotonic()
    attempt = 0
    while True:
        resp = requests.request(
            method,
            url,
            headers=headers,
            data=data,
            timeout=timeout,
            auth=auth,
        )
        if resp.status_code != 429:
            return resp
        attempt += 1
        elapsed = time.monotonic() - start
        if elapsed >= max_retry_seconds:
            log_task_event(
                f"Web retrieval rate limit from {label}; giving up after {attempt} attempts"
            )
            raise RetrievalRateLimitError(label=label, attempts=attempt, elapsed=elapsed)
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            sleep_for = int(retry_after)
        else:
            sleep_for = min(RETRY_MAX_SLEEP, RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
        remaining = max_retry_seconds - elapsed
        if sleep_for > remaining:
            sleep_for = remaining
        if sleep_for > 0:
            log_task_event(
                f"Web retrieval rate limit from {label}; retrying (attempt {attempt}) in {sleep_for}s"
            )
            time.sleep(sleep_for)


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return _clean_text(html.unescape(text))


def fetch_url_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="fetch_url")
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        return _strip_html(resp.text)
    return _clean_text(resp.text)


def _semantic_rerank(
    query: str,
    results: list[WebResult],
    config: EmbeddingConfig,
) -> list[WebResult]:
    if not results:
        return results
    texts = [query]
    for result in results:
        text = result.text or ""
        if len(text) > 2000:
            text = text[:2000]
        texts.append(f"{result.title}\n{text}".strip())
    client = EmbeddingClient(config)
    embeddings = client.embed_texts(texts)
    if len(embeddings) != len(texts):
        return results
    query_vec = embeddings[0]
    scored = []
    for idx, result in enumerate(results, start=1):
        sim = cosine_similarity(query_vec, embeddings[idx])
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
    return queries


def search_duckduckgo(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> list[WebResult]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": USER_AGENT}
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="duckduckgo")
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


def search_brave(
    query: str,
    api_key: str,
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[WebResult]:
    q = quote_plus(query)
    url = f"https://api.search.brave.com/res/v1/web/search?q={q}&count={max_results}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="brave")
    resp.raise_for_status()
    data = resp.json()
    results: list[WebResult] = []
    web_block = data.get("web") or {}
    items = web_block.get("results") or data.get("results") or []
    for item in items:
        title = _clean_text(item.get("title") or "")
        url = item.get("url") or ""
        snippet = item.get("description") or item.get("snippet") or item.get("content") or ""
        snippet = _clean_text(snippet)
        if not url:
            continue
        results.append(WebResult(title=title or url, url=url, text=snippet, source="brave"))
    return results


def search_google_pse(
    query: str,
    api_key: str,
    cx: str,
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[WebResult]:
    q = quote_plus(query)
    num = max(1, min(10, max_results))
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={q}&num={num}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="google_pse")
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items") or []
    results: list[WebResult] = []
    for item in items:
        title = _clean_text(item.get("title") or "")
        url = item.get("link") or ""
        snippet = _clean_text(item.get("snippet") or "")
        if not url:
            continue
        results.append(WebResult(title=title or url, url=url, text=snippet, source="google_pse"))
    return results


def search_tavily(
    query: str,
    api_key: str,
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[WebResult]:
    url = "https://api.tavily.com/search"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "query": query,
        "max_results": max(0, min(20, max_results)),
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    resp = _request_with_retries(
        "POST",
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
        label="tavily",
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("results") or []
    results: list[WebResult] = []
    for item in items:
        title = _clean_text(item.get("title") or "")
        url = item.get("url") or ""
        content = _clean_text(item.get("content") or "")
        if not url:
            continue
        results.append(WebResult(title=title or url, url=url, text=content, source="tavily"))
    return results


def _get_reddit_token(
    client_id: str,
    client_secret: str,
    user_agent: str,
    timeout: int,
) -> str | None:
    global _REDDIT_TOKEN, _REDDIT_TOKEN_EXPIRES_AT
    now = time.time()
    if _REDDIT_TOKEN and now < _REDDIT_TOKEN_EXPIRES_AT - 30:
        return _REDDIT_TOKEN
    url = "https://www.reddit.com/api/v1/access_token"
    headers = {"User-Agent": user_agent}
    data = {"grant_type": "client_credentials"}
    resp = _request_with_retries(
        "POST",
        url,
        headers=headers,
        data=data,
        timeout=timeout,
        auth=(client_id, client_secret),
        label="reddit_token",
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("access_token")
    expires_in = payload.get("expires_in") or 0
    try:
        expires_in = float(expires_in)
    except (TypeError, ValueError):
        expires_in = 0
    if token:
        _REDDIT_TOKEN = token
        _REDDIT_TOKEN_EXPIRES_AT = now + max(0, expires_in)
        return token
    return None


def search_reddit(
    query: str,
    client_id: str,
    client_secret: str,
    user_agent: str,
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[WebResult]:
    token = _get_reddit_token(client_id, client_secret, user_agent, timeout)
    if not token:
        return []
    limit = max(1, min(100, max_results))
    q = quote_plus(query)
    url = (
        "https://oauth.reddit.com/search"
        f"?q={q}&limit={limit}&sort=relevance&t=all&type=link"
    )
    headers = {"User-Agent": user_agent, "Authorization": f"bearer {token}"}
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="reddit")
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("data", {}).get("children", []) if isinstance(payload, dict) else []
    results: list[WebResult] = []
    for item in items:
        data = item.get("data", {}) if isinstance(item, dict) else {}
        title = _clean_text(data.get("title") or "")
        text = _clean_text(data.get("selftext") or "")
        permalink = data.get("permalink") or ""
        url = f"https://www.reddit.com{permalink}" if permalink else (data.get("url") or "")
        if not url:
            continue
        results.append(WebResult(title=title or url, url=url, text=text, source="reddit"))
        if len(results) >= max_results:
            break
    return results


def search_x_recent(
    query: str,
    bearer_token: str,
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[WebResult]:
    q = quote_plus(query)
    request_max = max(10, min(100, max_results))
    url = f"https://api.x.com/2/tweets/search/recent?query={q}&max_results={request_max}"
    headers = {"User-Agent": USER_AGENT, "Authorization": f"Bearer {bearer_token}"}
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="x")
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("data") or []
    results: list[WebResult] = []
    for item in items:
        tweet_id = item.get("id") or ""
        text = _clean_text(item.get("text") or "")
        if not tweet_id:
            continue
        url = f"https://x.com/i/web/status/{tweet_id}"
        results.append(WebResult(title=text or url, url=url, text=text, source="x"))
        if len(results) >= max_results:
            break
    return results


def search_arxiv(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> list[WebResult]:
    q = quote_plus(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}"
    headers = {"User-Agent": USER_AGENT}
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="arxiv")
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
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="semantic_scholar")
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
    resp = _request_with_retries("GET", url, headers=headers, timeout=timeout, label="crossref")
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
    if "tech" in modes_set:
        modes_set.add("open")
    open_providers: list[tuple[str, callable]] = []
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if brave_key:
        open_providers.append(
            ("brave", lambda q, max_results, timeout: search_brave(q, brave_key, max_results, timeout))
        )
    google_pse_key = os.getenv("GOOGLE_PSE_API_KEY")
    google_pse_cx = os.getenv("GOOGLE_PSE_CX")
    if google_pse_key and google_pse_cx:
        open_providers.append(
            (
                "google_pse",
                lambda q, max_results, timeout: search_google_pse(
                    q, google_pse_key, google_pse_cx, max_results, timeout
                ),
            )
        )
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        open_providers.append(
            ("tavily", lambda q, max_results, timeout: search_tavily(q, tavily_key, max_results, timeout))
        )

    forum_providers: list[tuple[str, callable]] = []
    x_token = os.getenv("X_API_BEARER_TOKEN")
    if x_token:
        forum_providers.append(
            ("x", lambda q, max_results, timeout: search_x_recent(q, x_token, max_results, timeout))
        )
    reddit_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT")
    if reddit_id and reddit_secret and reddit_user_agent:
        forum_providers.append(
            (
                "reddit",
                lambda q, max_results, timeout: search_reddit(
                    q, reddit_id, reddit_secret, reddit_user_agent, max_results, timeout
                ),
            )
        )
    log_task_event(
        "Web retrieval start: "
        f"query='{query}' "
        f"modes={sorted(modes_set) if modes_set else []} "
        f"max_results={max_results} "
        f"timeout={timeout}s "
        f"fetch_pages={fetch_pages} "
        f"hybrid={hybrid} "
        f"expand={expand_query_flag} "
        f"max_queries={max_queries} "
        f"open_sources={[label for label, _ in open_providers]} "
        f"forum_sources={[label for label, _ in forum_providers]}"
    )

    def _safe_search(label: str, fn: callable, query_text: str) -> list[WebResult]:
        try:
            items = fn(query_text, max_results=max_results, timeout=timeout)
            log_task_event(f"Success: source={label} query='{query_text}' results={len(items)}")
            return items
        except Exception as exc:
            log_task_event(f"Failure: source={label} query='{query_text}' error={exc}")
            return []

    queries = [query]
    if expand_query_flag:
        queries = expand_queries(query, modes_set, max_queries=max_queries, extra_queries=extra_queries)
    if not queries:
        return results
    if len(queries) > 1:
        log_task_event(f"Web retrieval expanded queries: {queries}")

    open_enabled = "open" in modes_set or "web" in modes_set
    forum_enabled = "forum" in modes_set
    if open_enabled and not open_providers:
        log_task_event("Web retrieval open sources skipped: no API keys configured.")
    if forum_enabled and not forum_providers:
        log_task_event("Web retrieval forum sources skipped: no API keys configured.")

    for q in queries:
        if open_enabled and open_providers:
            for label, fn in open_providers:
                results.extend(_safe_search(label, fn, q))
        if forum_enabled and forum_providers:
            for label, fn in forum_providers:
                results.extend(_safe_search(label, fn, q))

        if "lit" in modes_set or "literature" in modes_set:
            results.extend(_safe_search("arxiv", search_arxiv, q))
            results.extend(_safe_search("semantic_scholar", search_semantic_scholar, q))
            results.extend(_safe_search("crossref", search_crossref, q))

    fetched: list[WebResult] = []
    if fetch_pages and results:
        log_task_event(f"Web retrieval fetch pages: total={len(results)}")
    fetch_failures = 0
    fetch_started = time.monotonic()
    for result in results:
        if not fetch_pages or result.text:
            fetched.append(result)
            continue
        try:
            text = fetch_url_text(result.url, timeout=timeout)
            fetched.append(WebResult(title=result.title, url=result.url, text=text, source=result.source))
            time.sleep(0.2)
        except Exception:
            fetch_failures += 1
            fetched.append(result)
    if fetch_pages and results:
        fetch_elapsed = time.monotonic() - fetch_started
        log_task_event(
            "Web retrieval fetch done: "
            f"fetched={len(fetched)} "
            f"failed={fetch_failures} "
            f"elapsed={fetch_elapsed:.1f}s"
        )
    merged = _dedupe_results(fetched)
    if not hybrid:
        return merged

    provider = (embed_provider or os.getenv("MINI_NEXEN_EMBED_PROVIDER") or "lmstudio").lower()
    model = embed_model or os.getenv("MINI_NEXEN_EMBED_MODEL")
    base_url = embed_base_url or os.getenv("MINI_NEXEN_EMBED_BASE_URL") or os.getenv("LMSTUDIO_BASE_URL")
    api_key = embed_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider != "gemini" and not base_url:
        return merged

    config = EmbeddingConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        timeout=embed_timeout or timeout,
        api_key=api_key,
    )
    try:
        log_task_event(
            "Web retrieval rerank start: "
            f"provider={config.provider} "
            f"model={config.model} "
            f"timeout={config.timeout}s "
            f"items={len(merged)}"
        )
        rerank_started = time.monotonic()
        reranked = _semantic_rerank(query=query, results=merged, config=config)
        rerank_elapsed = time.monotonic() - rerank_started
        log_task_event(
            "Web retrieval rerank done: "
            f"elapsed={rerank_elapsed:.1f}s "
            f"items={len(reranked)}"
        )
        return reranked
    except Exception as exc:
        log_task_event(f"Web retrieval rerank failed; returning original results. error={exc}")
        return merged
