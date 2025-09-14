
import os, tempfile, requests, fitz, trafilatura, arxiv
from typing import Any, Dict, List, Optional

# MCP server
from mcp.server import Server

# 可选：Tavily / GitHub
from tavily import TavilyClient

SERVER_NAME = "research"
server = Server(SERVER_NAME)

# ---------- 工具实现 ----------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def _web_search(query: str, max_results: int = 5, topic: str = "general") -> List[Dict[str, Any]]:
    if not tavily:
        return [{"title": "No Tavily API key", "url": "", "content": ""}]
    params = {"query": query, "search_depth": "advanced", "max_results": max_results}
    if topic == "developer":
        params["topic"] = "developer"
    r = tavily.search(**params)
    return [
        {
            "title": it.get("title"),
            "url": it.get("url"),
            "content": it.get("content"),
            "score": it.get("score"),
        } for it in r.get("results", [])
    ]

def _arxiv_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    s = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    out = []
    for r in s.results():
        out.append({
            "title": r.title,
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
            "published": r.published.strftime("%Y-%m-%d") if r.published else None,
            "authors": [a.name for a in r.authors],
            "summary": r.summary,
        })
    return out

def _github_code_search(query: str, language: Optional[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
    assert GITHUB_TOKEN, "Missing GITHUB_TOKEN"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2025-11-28"
    }
    q = query + (f" language:{language}" if language else "")
    url = f"https://api.github.com/search/code?q={requests.utils.quote(q)}&per_page={max_results}"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "name": it.get("name"),
            "path": it.get("path"),
            "repo": it.get("repository", {}).get("full_name"),
            "html_url": it.get("html_url"),
            "score": it.get("score"),
        } for it in data.get("items", [])
    ]

def _fetch_text(url: str) -> Dict[str, Any]:
    downloaded = trafilatura.fetch_url(url, no_ssl=True)
    text = trafilatura.extract(downloaded, include_comments=False, include_images=False) if downloaded else ""
    return {"url": url, "text": text or ""}

def _pdf_to_text(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(r.content)
        temp = f.name
    try:
        doc = fitz.open(temp)
        texts = [p.get_text("text") for p in doc]
        return {"url": url, "text": "\n".join(texts)[:200000]}
    finally:
        try: os.remove(temp)
        except: pass


