# mcp-a2a-agent


一、架构设计
- 前端（Next.js 14）
  - 聊天界面（支持引用链接渲染、代码高亮、流式）
- 后端（FastAPI）
  - 会话/历史管理、流式输出、用户请求→CrewAI
  - 在进程内通过 MCP Stdio 启动/连接各个 MCP Server
- 编排（CrewAI）
  - 单/多 Agent 策略（Planner/Researcher/Dev），按需求扩展
- MCP 工具层（Python MCP Server）
  - research server：web_search、arxiv、GitHub code search、fetch_text、pdf_to_text（可再拆分）
- 数据/模型
  - LLM: OpenAI gpt-4.1/4o/mini（或自换）
  - 向量库：先略（可加 Qdrant/Chroma + reranker）
- 流程
  - 前端请求 → 后端创建 Crew 调度 → 通过 MCP 调工具 → 汇总带引用 → 流式返回

二、项目结构
```
ai-agent-mcp-crewai/
├─ backend/
│  ├─ app/
│  │  ├─ main.py               # FastAPI 入口（/chat）
│  │  └─ settings.py           # 环境配置
│  ├─ agents/
│  │  ├─ crew.py               # Crew 定义（Agent/Task/Tools）
│  │  └─ mcp_tools.py          # MCP 客户端封装为 CrewAI Tool
│  ├─ mcp_servers/
│  │  └─ research/
│  │     ├─ __init__.py
│  │     └─ server.py          # MCP Research Server（工具集合）
│  └─ requirements.txt
├─ frontend/
│  ├─ package.json
│  ├─ next.config.js
│  └─ app/
│     └─ page.tsx              # 聊天页（调用后端 /chat）
├─ .env.example
└─ README.md
```



六、运行步骤
- 准备环境变量（.env）
  - OPENAI_API_KEY=...
  - TAVILY_API_KEY=...（可选，启用 web search）
  - GITHUB_TOKEN=...（可选，启用 GitHub 搜索）
- 安装并启动后端
  - cd backend
  - pip install -r requirements.txt
  - uvicorn app.main:app --reload --port 8000
- 启动前端
  - cd frontend
  - export NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
  - npm i && npm run dev
- 浏览器打开 http://localhost:3000

七、todo
- 多 Agent（CrewAI）
  - Planner（分解查询/路由）→ Researcher（学术）→ Dev（开发搜索+官方文档）→ Editor（汇总+引用）
- 向量库与上传
  - 增加 mcp-server-vector（Qdrant/Chroma），tools: vector_upsert/vector_query；后端加 /upload
- 重排与证据高亮
  - Reranker（Voyage/Cohere）+ 证据句提取（quote 对齐）
- 流式输出
  - FastAPI 使用 Server-Sent Events 或 WebSocket；前端逐 token 渲染
- 更多 MCP Server
  - 浏览器（Playwright/Firecrawl）、Slack/Jira、Postgres、AWS/GCP（通过 MCP 统一接入）

