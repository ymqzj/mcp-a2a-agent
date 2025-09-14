import os, json
from typing import Dict
from crewai import Agent, Task, Crew, Process
from crewai.tools.base_tool import BaseTool
from openai import OpenAI
import os

# Fallback to environment variables if app.settings is not available
class _Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-high")

settings = _Settings()
try:
    # prefer package-style import when run as part of project
    from backend.agents.mcp_tools import call_mcp_sync
except Exception:
    # fallback for direct script execution
    from mcp_tools import call_mcp_sync

class MCPCallTool(BaseTool):
    name: str = "mcp_tool"
    description: str = "Call a tool via MCP server"
    server_cmd: list = None
    tool_name: str = None
    env: Dict[str, str] = None

    def _run(self, **kwargs) -> str:
        result = call_mcp_sync(self.server_cmd, self.tool_name, kwargs, env=self.env)
        return json.dumps(result, ensure_ascii=False)

def make_tools():
    env_research = {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", ""),
    }
    env_vector = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        "EMBED_DIM": os.getenv("EMBED_DIM", "1536"),
        "QDRANT_URL": os.getenv("QDRANT_URL", "http://qdrant:6333"),
    }
    cmd_research = [os.getenv("PYTHON", "python"), "-m", "mcp_servers.research.server"]
    cmd_vector = [os.getenv("PYTHON", "python"), "-m", "mcp_servers.vector.server"]

    return dict(
        web_search=MCPCallTool(name="web_search", description="Web 搜索", server_cmd=cmd_research, tool_name="web_search", env=env_research),
        arxiv_search=MCPCallTool(name="arxiv_search", description="arXiv 检索", server_cmd=cmd_research, tool_name="arxiv_search", env=env_research),
        github_code_search=MCPCallTool(name="github_code_search", description="GitHub 代码搜索", server_cmd=cmd_research, tool_name="github_code_search", env=env_research),
        fetch_text=MCPCallTool(name="fetch_text", description="抓取网页正文", server_cmd=cmd_research, tool_name="fetch_text", env=env_research),
        pdf_to_text=MCPCallTool(name="pdf_to_text", description="PDF 解析", server_cmd=cmd_research, tool_name="pdf_to_text", env=env_research),
        vector_query=MCPCallTool(name="vector_query", description="向量检索（Qdrant）", server_cmd=cmd_vector, tool_name="vector_query", env=env_vector),
        vector_upsert=MCPCallTool(name="vector_upsert", description="向量写入（Qdrant）", server_cmd=cmd_vector, tool_name="vector_upsert", env=env_vector),
    )

def build_crew():
    tools = make_tools()
    llm = OpenAI(api_key=settings.OPENAI_API_KEY, default_model=settings.OPENAI_MODEL)

    planner = Agent(
        role="Planner",
        goal="分解用户问题为检索/开发子任务并确定工具路线",
        backstory="严谨的研究助理，擅长把开放问题拆解为可执行步骤。",
        llm=llm, verbose=False, tools=[], allow_delegation=False,
    )

    researcher = Agent(
        role="Researcher",
        goal="执行学术与网页检索、抓取正文/PDF，总结证据并保留引用",
        backstory="擅长 arXiv/官网文档/高质量技术博客的证据收集。",
        llm=llm, verbose=False,
        tools=[tools["web_search"], tools["arxiv_search"], tools["fetch_text"], tools["pdf_to_text"], tools["vector_query"]],
        allow_delegation=False,
    )

    dev = Agent(
        role="DevSearch",
        goal="执行 GitHub 代码与官方文档检索，返回实现参考与链接",
        backstory="熟悉 GitHub 搜索语法与工程最佳实践。",
        llm=llm, verbose=False,
        tools=[tools["github_code_search"], tools["fetch_text"]],
        allow_delegation=False,
    )

    editor = Agent(
        role="Editor",
        goal="把所有证据整合为中文回答，逐点附带可点击引用（标题/链接/日期）",
        backstory="对引用溯源和不确定性声明很严格。",
        llm=llm, verbose=False, tools=[], allow_delegation=False,
    )

    plan_task = Task(
        description="根据用户问题制定检索/开发搜索计划，输出小列表，含：检索式、工具选择、判定标准。",
        agent=planner, expected_output="结构化计划（列表）"
    )
    research_task = Task(
        description="按计划执行学术/网页检索和抓取，必要时阅读 PDF；输出关键结论+引用。",
        agent=researcher, expected_output="研究笔记（带引用）", context=[plan_task]
    )
    dev_task = Task(
        description="执行 GitHub/官方文档检索，给出实现示例/仓库/API 文档链接。",
        agent=dev, expected_output="开发参考（带链接）", context=[plan_task]
    )
    synth_task = Task(
        description="综合前述输出，形成最终回答（中文），每个结论点附 [1][2]... 引用。标注不确定性。",
        agent=editor, expected_output="最终回答", context=[plan_task, research_task, dev_task]
    )

    crew = Crew(agents=[planner, researcher, dev, editor], tasks=[plan_task, research_task, dev_task, synth_task], process=Process.sequential)
    return crew

def answer_question(question: str) -> str:
    crew = build_crew()
    result = crew.kickoff(inputs={"question": question})
    return str(result)
