下面给你一套可以直接开工的方案：从最小可用版（MVP）到可进阶的生产级架构，覆盖 ChatGPT 风格对话、文献检索、开发搜索（GitHub/官方文档/StackOverflow）与网页浏览总结。附带可运行的 Python 参考实现（函数调用式 Agent）。

概览
- 能力目标
  - 对话式问答（中文优先）
  - 学术检索与可追溯引用（arXiv / OpenAlex / Crossref / Semantic Scholar / PubMed）
  - 开发搜索（GitHub code search / Stack Overflow / 官方文档）
  - 网页抓取、PDF 解析、摘要与对比
  - 基于向量库的本地文档检索（RAG），含重排与去重
- 推荐栈
  - LLM: OpenAI gpt-4.1 / gpt-4o / gpt-4.1-mini
  - Embedding: text-embedding-3-large/small
  - 搜索/数据源: Tavily/Bing Web Search、arXiv、OpenAlex、Semantic Scholar、GitHub API
  - 抓取/解析: Firecrawl/Playwright 或 trafilatura、PyMuPDF
  - 向量库: Chroma/FAISS；重排: Cohere Rerank / Voyage Rerank（可选）
  - 编排: OpenAI function calling（或 LangChain / LlamaIndex / Semantic Kernel）

MVP 架构（单 Agent + 工具）
- Orchestrator: LLM 作为 Planner/Executor（通过 function calling 调工具）
- 工具集
  - web_search: 通用/开发类网页搜索（Tavily/Bing）
  - academic_search: arXiv/OpenAlex/Semantic Scholar
  - github_code_search: GitHub 代码搜索
  - fetch_and_parse: 抓取网页正文（去广告/导航）
  - pdf_to_text: PDF 解析
  - vector_upsert/vector_query: 本地向量检索（可选）
- 策略
  - Query rewrite & multi-step：先改写检索式，再多步搜索→抓取→去重→重排→综合
  - 生成时强制引用来源并附上链接与发布时间
  - 置信度与覆盖率提示（当来源少或矛盾时）

一键起步（Python 参考实现）
依赖
- Python 3.10+
- pip install:
  - openai tavily-python arxiv trafilatura pymupdf beautifulsoup4 requests chromadb tenacity pydantic

环境变量
- OPENAI_API_KEY
- TAVILY_API_KEY（如用 Tavily）
- GITHUB_TOKEN（如启用 GitHub code search）

代码（agent.py）
```python
import os
import json
import time
import tempfile
from typing import List, Dict, Any, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

# LLM
from openai import OpenAI
# Web search
from tavily import TavilyClient
# Academic
import arxiv
# Content extraction
import trafilatura
# PDF parsing
import fitz  # PyMuPDF
# Vector store (optional)
import chromadb

# ======================
# Config
# ======================

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # 质量/成本平衡
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Vector DB (in-memory; switch to persist_dir for production)
chroma_client = chromadb.Client()
collection = chromadb.Client().create_collection(
    name="agent_docs",
    metadata={"hnsw:space": "cosine"}
)

# ======================
# Utilities
# ======================

def embed_texts(texts: List[str]) -> List[List[float]]:
    chunks = [texts[i:i+100] for i in range(0, len(texts), 100)]
    vectors = []
    for batch in chunks:
        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        vectors.extend([d.embedding for d in emb.data])
    return vectors

def vector_upsert(docs: List[Dict[str, Any]]):
    # docs: [{id, text, metadata}]
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d.get("metadata", {}) for d in docs]
    embs = embed_texts(texts)
    collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

def vector_query(query: str, k: int = 5):
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i]
        })
    return hits

# ======================
# Tools
# ======================

def tool_web_search(query: str, max_results: int = 5, topic: str = "general") -> List[Dict[str, Any]]:
    """
    topic: general / developer
    """
    results = []
    if tavily:
        params = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
        }
        if topic == "developer":
            params["topic"] = "developer"
        r = tavily.search(**params)
        for item in r.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score")
            })
    else:
        # fallback: simple DuckDuckGo-lite via html (not recommended for prod)
        # Encourage using Tavily/Bing API in production
        pass
    return results

def tool_academic_search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    out = []
    for r in search.results():
        out.append({
            "title": r.title,
            "url": r.entry_id,
            "published": r.published.strftime("%Y-%m-%d") if r.published else None,
            "summary": r.summary,
            "authors": [a.name for a in r.authors],
            "pdf_url": r.pdf_url
        })
    return out

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def tool_github_code_search(query: str, language: Optional[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
    assert GITHUB_TOKEN, "GITHUB_TOKEN is required for GitHub code search."
    headers = {
        "Accept": "application/vnd.github.text-match+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    q = query
    if language:
        q += f" language:{language}"
    url = f"https://api.github.com/search/code?q={requests.utils.quote(q)}&per_page={max_results}"
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for item in data.get("items", []):
        out.append({
            "name": item.get("name"),
            "path": item.get("path"),
            "repo": item.get("repository", {}).get("full_name"),
            "html_url": item.get("html_url"),
            "score": item.get("score")
        })
    return out

def tool_fetch_and_parse(url: str) -> Dict[str, Any]:
    downloaded = trafilatura.fetch_url(url, no_ssl=True)
    text = trafilatura.extract(downloaded, include_comments=False, include_images=False) if downloaded else None
    return {"url": url, "text": text or ""}

def tool_pdf_to_text(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(r.content)
        temp_path = f.name
    try:
        doc = fitz.open(temp_path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        full = "\n".join(texts)
        return {"url": url, "text": full[:200000]}  # trim to avoid oversized payload
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

def tool_vector_upsert(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    vector_upsert(docs)
    return {"ok": True, "count": len(docs)}

def tool_vector_query(query: str, k: int = 5) -> List[Dict[str, Any]]:
    return vector_query(query, k=k)

# ======================
# Tool Schemas for function calling
# ======================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for general or developer topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                    "topic": {"type": "string", "enum": ["general", "developer"], "default": "general"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "academic_search_arxiv",
            "description": "Search academic papers on arXiv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "github_code_search",
            "description": "Search code on GitHub.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "language": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_and_parse",
            "description": "Fetch a URL and extract main readable text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pdf_to_text",
            "description": "Download a PDF and extract text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_upsert",
            "description": "Upsert documents into local vector DB for RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "metadata": {"type": "object"}
                            },
                            "required": ["id", "text"]
                        }
                    }
                },
                "required": ["docs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_query",
            "description": "Query local vector DB with semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_MAP = {
    "web_search": lambda args: tool_web_search(**args),
    "academic_search_arxiv": lambda args: tool_academic_search_arxiv(**args),
    "github_code_search": lambda args: tool_github_code_search(**args),
    "fetch_and_parse": lambda args: tool_fetch_and_parse(**args),
    "pdf_to_text": lambda args: tool_pdf_to_text(**args),
    "vector_upsert": lambda args: tool_vector_upsert(**args["docs"]),
    "vector_query": lambda args: tool_vector_query(**args)
}

SYSTEM_PROMPT = """你是一个研究与开发搜索 Agent：
- 当问题需要外部信息时，优先调用搜索/抓取工具，充分检索后再回答。
- 先改写检索式（如加入同义词、英文关键词），再多源检索，去重与对比。
- 回答必须列出明确可点击的引用来源（[1], [2]...），包含标题、链接、日期（若有）。
- 若证据不足或来源冲突，清楚说明不确定性与建议的下一步检索方向。
- 技术问题优先查官方文档与源代码；学术问题优先查 arXiv/学术源。
- 中文输出，关键术语保留英文。
"""

def call_llm(messages: List[Dict[str, str]], tools=TOOLS):
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2
    )

def run_agent(user_query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    while True:
        resp = call_llm(messages)
        msg = resp.choices[0].message

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                # Run tool
                try:
                    result = TOOL_MAP[name](args)
                except Exception as e:
                    result = {"error": str(e)}
                # Append tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False)[:200000]
                })
            # Continue loop for next reasoning step
            continue
        else:
            return msg.content

if __name__ == "__main__":
    q = "请检索最近一年关于 LLM 代码检索（code retrieval）的论文进展，并给出关键方法对比与链接；另外找下 GitHub 上典型实现。"
    print(run_agent(q))
```

如何运行
- 安装依赖、配置环境变量后直接执行：
  - python agent.py
- 在聊天循环中，模型会按需调用搜索/抓取/解析工具，再输出带引用的综合结论。

进阶增强（生产建议）
- 多源学术接口
  - OpenAlex、Crossref、Semantic Scholar、PubMed（并行检索→合并→去重→重排）
- 重排与聚类
  - 引入 reranker（Cohere Rerank / Voyage）提升前列质量
  - 基于 URL 主域+标题+embedding 去重，语义聚类（比如 HDBSCAN）做主题分组
- 多跳检索与查询分解
  - “主题→子问题→证据收集→综合”，对综述/对比类问题尤其有效
- 引用与可追溯性
  - 强制每个结论段落关联证据句（quote/line range）
  - 当仅有二手来源时标注“二手引用”
- PDF 与本地库
  - 批量 PDF 下载→PDF-to-Text→段落切分→embedding→Chroma/FAISS
  - 支持“上传 PDF/Markdown/代码仓库”并与在线检索融合（hybrid RAG）
- Hallucination Guard
  - 只允许从检索证据中生成结论；若无证据则说明“无可靠来源”
- 缓存与速率
  - Requests 级别缓存（diskcache/httpx-cache）、搜索结果 KV 缓存
  - 并发与退避重试（tenacity 已示例）
- 观测与评估
  - 日志与指标（命中率、NDCG@k、首文档时间）
  - 任务集回放（LLM-as-a-judge+人工抽查）
- UI 与部署
  - FastAPI/Flask 暴露 REST；前端 Next.js/Streamlit
  - 队列（Celery/Arq）执行重抓取任务；向量库持久化（Chroma persist_dir / pgvector）
  - 监控与告警（OpenTelemetry + Prometheus/Grafana）

可选的数据源与工具对照
- 学术
  - arXiv: 预印本，免费，覆盖强；无 DOI 元数据时可用 Crossref 补全
  - OpenAlex: 开放学术图谱，适合元数据/引用网络
  - Semantic Scholar: 摘要/影响力指标好，用作补充
  - PubMed: 医学/生物方向
- 开发
  - GitHub: 代码搜索与仓库 issue/PR
  - Stack Overflow API: 问答质量好
  - 官方文档: MDN、Python docs、PyTorch/TF、K8s、AWS/GCP/Azure
- 通用搜索
  - Tavily: 开发向/学术向优化，简单易用
  - Bing/Brave: 商用可用性与覆盖较稳
