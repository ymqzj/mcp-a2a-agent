
from mcp import mcp_client
from typing import List, Dict
import json
import asyncio
from openai import AsyncOpenAI
class ChatOpenAI:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.client = AsyncOpenAI()

    async def ainvoke(self, prompt):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return type("Resp", (), {"content": response.choices[0].message.content})()

class MCPSearchAgent:
    def __init__(self, server_url: str = "http://localhost:3000"):
        self.client = mcp_client.client()
        self.server_url = server_url
        self.llm = ChatOpenAI(model="gpt-4")

    async def connect(self):
        print(f"Connecting to MCP server at {self.server_url} ...")
        await self.client.connect(self.server_url)
        self.tools = await self.client.list_tools()
        print(f"可用工具: {[tool.name for tool in self.tools]}")

    async def search_and_answer(self, user_query: str) -> str:
        # 1. 分析查询意图
        intent = await self._analyze_query(user_query)
        print(f"分析意图: {intent}")

        # 2. 调用搜索工具
        search_results = await self.client.call_tool(
            "web_search",
            {"query": intent["search_query"], "num_results": 5}
        )
        print(f"搜索结果: {search_results}")

        # 3. 提取网页内容
        urls = [r["url"] for r in search_results["results"][:3]]
        contents = []
        for url in urls:
            result = await self.client.call_tool(
                "extract_webpage",
                {"url": url}
            )
            contents.append(result["content"])
        print(f"网页内容: {contents}")

        # 4. 语义搜索相关段落
        relevant_chunks = await self.client.call_tool(
            "semantic_search",
            {
                "query": user_query,
                "documents": contents,
                "top_k": 5
            }
        )
        print(f"相关段落: {relevant_chunks}")

        # 5. 生成最终答案
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        answer = await self._generate_answer(user_query, context)
        return answer

    async def _analyze_query(self, query: str) -> Dict:
        prompt = f"分析以下查询，提取搜索关键词：\n用户查询: {query}\n\n返回JSON格式：\n{{\"search_query\": \"优化后的搜索关键词\", \"intent\": \"查询意图\"}}"
        response = await self.llm.ainvoke(prompt)
        # 真实 LLM 返回应为 JSON 格式
        try:
            return json.loads(response.content)
        except Exception:
            # fallback
            return {"search_query": query, "intent": "search"}

    async def _generate_answer(self, user_query, context):
        # 可接入真实 LLM
        return f"Q: {user_query}\nA: {context}"

async def main():
    agent = MCPSearchAgent()
    await agent.connect()
    user_query = "什么是MCP协议？"
    answer = await agent.search_and_answer(user_query)
    print("\n最终答案：\n", answer)

if __name__ == "__main__":
    asyncio.run(main())
