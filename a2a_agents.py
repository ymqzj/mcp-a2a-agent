import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List
import uuid

@dataclass
class Message:
    sender: str
    receiver: str
    content: Any
    msg_type: str  # "request", "response", "broadcast"
    correlation_id: str

class BaseAgent(ABC):
    def __init__(self, name: str, message_bus: 'MessageBus'):
        self.name = name
        self.message_bus = message_bus
        self.inbox = asyncio.Queue()
    @abstractmethod
    async def process_message(self, message: Message) -> Any:
        pass
    async def send_message(self, receiver: str, content: Any, msg_type: str = "request"):
        message = Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            correlation_id=str(uuid.uuid4())
        )
        await self.message_bus.route_message(message)
    async def run(self):
        while True:
            message = await self.inbox.get()
            response = await self.process_message(message)
            if response and message.msg_type == "request":
                await self.send_message(
                    message.sender,
                    response,
                    "response"
                )

# 伪实现依赖
class SearchEngine:
    async def search_bing(self, query):
        return [{"url": f"https://google.com?q={query}", "title": f"Result {i+1}"} for i in range(5)]
class ContentExtractor:
    async def batch_extract(self, urls):
        return [f"内容提取自 {url}" for url in urls]
class VectorStore:
    def add_texts(self, texts):
        self.texts = texts
    def search(self, query, k=5):
        return self.texts[:k]
class ChatOpenAI:
    def __init__(self, model="gpt-4"): self.model = model
    async def ainvoke(self, prompt):
        return type("Resp", (), {"content": f"AI回答: {prompt}"})()

class SearchAgent(BaseAgent):
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.search_engine = SearchEngine()
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "search":
            query = message.content["query"]
            results = await self.search_engine.search_bing(query)
            return {"results": results}
class ContentAgent(BaseAgent):
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.extractor = ContentExtractor()
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "extract":
            urls = message.content["urls"]
            contents = await self.extractor.batch_extract(urls)
            return {"contents": contents}
class AnalysisAgent(BaseAgent):
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.vector_store = VectorStore()
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "analyze":
            texts = message.content["texts"]
            query = message.content["query"]
            self.vector_store.add_texts(texts)
            relevant = self.vector_store.search(query, k=5)
            return {"relevant_chunks": relevant}
class SynthesisAgent(BaseAgent):
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.llm = ChatOpenAI(model="gpt-4")
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "synthesize":
            query = message.content["query"]
            context = message.content["context"]
            prompt = f"""基于以下信息回答问题：\n问题: {query}\n相关信息:\n{context}\n请提供准确、全面的答案。"""
            response = await self.llm.ainvoke(prompt)
            return {"answer": response.content}

class MessageBus:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
    async def route_message(self, message: Message):
        if message.receiver in self.agents:
            await self.agents[message.receiver].inbox.put(message)
        elif message.receiver == "broadcast":
            for agent in self.agents.values():
                if agent.name != message.sender:
                    await agent.inbox.put(message)

class OrchestratorAgent(BaseAgent):
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.workflows = {}
    async def process_message(self, message: Message) -> Any:
        # OrchestratorAgent 主要通过 handle_user_query 协调，不处理普通消息
        return None
    async def handle_user_query(self, query: str) -> str:
        workflow_id = str(uuid.uuid4())
        await self.send_message(
            "search_agent",
            {"action": "search", "query": query, "workflow_id": workflow_id}
        )
        search_results = await self._wait_for_response("search_agent")
        urls = [r["url"] for r in search_results["results"][:3]]
        await self.send_message(
            "content_agent",
            {"action": "extract", "urls": urls, "workflow_id": workflow_id}
        )
        content_results = await self._wait_for_response("content_agent")
        await self.send_message(
            "analysis_agent",
            {
                "action": "analyze",
                "texts": content_results["contents"],
                "query": query,
                "workflow_id": workflow_id
            }
        )
        analysis_results = await self._wait_for_response("analysis_agent")
        await self.send_message(
            "synthesis_agent",
            {
                "action": "synthesize",
                "query": query,
                "context": "\n\n".join(analysis_results["relevant_chunks"]),
                "workflow_id": workflow_id
            }
        )
        final_answer = await self._wait_for_response("synthesis_agent")
        return final_answer["answer"]
    async def _wait_for_response(self, agent_name):
        # 简单实现：直接取inbox
        return await self.inbox.get()
