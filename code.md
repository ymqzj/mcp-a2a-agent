
## 🔌 **MCP (Model Context Protocol) 架构**

MCP 是 Anthropic 最近推出的协议，让 AI 模型能够安全地与外部数据源和工具交互。

### MCP Server 实现（搜索服务端）

```python
# mcp_search_server.py
from mcp.server import Server, Tool
from mcp.server.models import CallToolRequest, CallToolResponse
import asyncio
from typing import Any, Dict

class SearchMCPServer:
    def __init__(self):
        self.server = Server("web-search-server")
        self._setup_tools()
    
    def _setup_tools(self):
        @self.server.tool()
        async def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
            """执行网络搜索"""
            # 这里接入实际的搜索API
            results = await self._perform_search(query, num_results)
            return {
                "results": results,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.server.tool()
        async def extract_webpage(url: str) -> Dict[str, Any]:
            """提取网页内容"""
            content = await self._extract_content(url)
            return {
                "url": url,
                "content": content,
                "word_count": len(content.split())
            }
        
        @self.server.tool()
        async def semantic_search(
            query: str, 
            documents: List[str], 
            top_k: int = 3
        ) -> List[Dict[str, Any]]:
            """在文档中进行语义搜索"""
            results = await self._semantic_search(query, documents, top_k)
            return results
    
    async def _perform_search(self, query: str, num_results: int):
        # 实际搜索逻辑
        async with aiohttp.ClientSession() as session:
            # Bing/Google API 调用
            pass
    
    def run(self, host="localhost", port=3000):
        """启动MCP服务器"""
        self.server.run(host=host, port=port)
```

### MCP Client 实现（Agent端）

```python
# mcp_search_client.py
from mcp.client import Client
from typing import List, Dict
import json

class MCPSearchAgent:
    def __init__(self, server_url: str = "http://localhost:3000"):
        self.client = Client()
        self.server_url = server_url
        self.llm = ChatOpenAI(model="gpt-4")
        
    async def connect(self):
        """连接到MCP服务器"""
        await self.client.connect(self.server_url)
        self.tools = await self.client.list_tools()
        print(f"可用工具: {[tool.name for tool in self.tools]}")
    
    async def search_and_answer(self, user_query: str) -> str:
        """使用MCP工具回答问题"""
        
        # 1. 分析查询意图
        intent = await self._analyze_query(user_query)
        
        # 2. 调用搜索工具
        search_results = await self.client.call_tool(
            "web_search",
            {"query": intent["search_query"], "num_results": 5}
        )
        
        # 3. 提取网页内容
        urls = [r["url"] for r in search_results["results"][:3]]
        contents = []
        for url in urls:
            result = await self.client.call_tool(
                "extract_webpage",
                {"url": url}
            )
            contents.append(result["content"])
        
        # 4. 语义搜索相关段落
        relevant_chunks = await self.client.call_tool(
            "semantic_search",
            {
                "query": user_query,
                "documents": contents,
                "top_k": 5
            }
        )
        
        # 5. 生成最终答案
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        answer = await self._generate_answer(user_query, context)
        
        return answer
    
    async def _analyze_query(self, query: str) -> Dict:
        """分析用户查询意图"""
        prompt = f"""分析以下查询，提取搜索关键词：
        用户查询: {query}
        
        返回JSON格式：
        {{"search_query": "优化后的搜索关键词", "intent": "查询意图"}}
        """
        
        response = await self.llm.ainvoke(prompt)
        return json.loads(response.content)
```

## 🤝 **A2A (Agent-to-Agent) 协作架构**

A2A 模式让多个专门的 Agent 协作完成复杂任务。

### Agent 定义

```python
# a2a_agents.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List
import asyncio

@dataclass
class Message:
    """Agent间通信消息"""
    sender: str
    receiver: str
    content: Any
    msg_type: str  # "request", "response", "broadcast"
    correlation_id: str

class BaseAgent(ABC):
    """基础Agent类"""
    def __init__(self, name: str, message_bus: 'MessageBus'):
        self.name = name
        self.message_bus = message_bus
        self.inbox = asyncio.Queue()
        
    @abstractmethod
    async def process_message(self, message: Message) -> Any:
        """处理接收到的消息"""
        pass
    
    async def send_message(self, receiver: str, content: Any, msg_type: str = "request"):
        """发送消息给其他Agent"""
        message = Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            correlation_id=str(uuid.uuid4())
        )
        await self.message_bus.route_message(message)
    
    async def run(self):
        """Agent主循环"""
        while True:
            message = await self.inbox.get()
            response = await self.process_message(message)
            if response and message.msg_type == "request":
                await self.send_message(
                    message.sender,
                    response,
                    "response"
                )
```

### 专门化的 Agents

```python
class SearchAgent(BaseAgent):
    """负责网络搜索的Agent"""
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.search_engine = SearchEngine()
    
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "search":
            query = message.content["query"]
            results = await self.search_engine.search_bing(query)
            return {"results": results}

class ContentAgent(BaseAgent):
    """负责内容提取的Agent"""
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.extractor = ContentExtractor()
    
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "extract":
            urls = message.content["urls"]
            contents = await self.extractor.batch_extract(urls)
            return {"contents": contents}

class AnalysisAgent(BaseAgent):
    """负责内容分析的Agent"""
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.vector_store = VectorStore()
    
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "analyze":
            texts = message.content["texts"]
            query = message.content["query"]
            
            # 向量化和检索
            self.vector_store.add_texts(texts)
            relevant = self.vector_store.search(query, k=5)
            
            return {"relevant_chunks": relevant}

class SynthesisAgent(BaseAgent):
    """负责生成最终答案的Agent"""
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.llm = ChatOpenAI(model="gpt-4")
    
    async def process_message(self, message: Message) -> Any:
        if message.content["action"] == "synthesize":
            query = message.content["query"]
            context = message.content["context"]
            
            prompt = f"""基于以下信息回答问题：
            问题: {query}
            
            相关信息:
            {context}
            
            请提供准确、全面的答案。"""
            
            response = await self.llm.ainvoke(prompt)
            return {"answer": response.content}
```

### 消息总线和协调器

```python
class MessageBus:
    """Agent间消息路由"""
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        self.agents[agent.name] = agent
    
    async def route_message(self, message: Message):
        """路由消息到目标Agent"""
        if message.receiver in self.agents:
            await self.agents[message.receiver].inbox.put(message)
        elif message.receiver == "broadcast":
            # 广播消息
            for agent in self.agents.values():
                if agent.name != message.sender:
                    await agent.inbox.put(message)

class OrchestratorAgent(BaseAgent):
    """协调器Agent，管理整个搜索流程"""
    def __init__(self, name: str, message_bus: 'MessageBus'):
        super().__init__(name, message_bus)
        self.workflows = {}
    
    async def handle_user_query(self, query: str) -> str:
        """处理用户查询的完整流程"""
        workflow_id = str(uuid.uuid4())
        
        # 1. 发起搜索
        await self.send_message(
            "search_agent",
            {"action": "search", "query": query, "workflow_id": workflow_id}
        )
        search_results = await self._wait_for_response("search_agent")
        
        # 2. 提取内容
        urls = [r["url"] for r in search_results["results"][:3]]
        await self.send_message(
            "content_agent",
            {"action": "extract", "urls": urls, "workflow_id": workflow_id}
        )
        content_results = await self._wait_for_response("content_agent")
        
        # 3. 分析内容
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
        
        # 4. 生成答案
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
```

### 完整的 A2A 系统启动

```python
class A2AWebSearchSystem:
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents = []
        self._setup_agents()
    
    def _setup_agents(self):
        """初始化所有Agent"""
        # 创建各个专门的Agent
        search_agent = SearchAgent("search_agent", self.message_bus)
        content_agent = ContentAgent("content_agent", self.message_bus)
        analysis_agent = AnalysisAgent("analysis_agent", self.message_bus)
        synthesis_agent = SynthesisAgent("synthesis_agent", self.message_bus)
        orchestrator = OrchestratorAgent("orchestrator", self.message_bus)
        
        # 注册到消息总线
        for agent in [search_agent, content_agent, analysis_agent, synthesis_agent, orchestrator]:
            self.message_bus.register_agent(agent)
            self.agents.append(agent)
    
    async def start(self):
        """启动所有Agent"""
        tasks = [agent.run() for agent in self.agents]
        await asyncio.gather(*tasks)
    
    async def query(self, user_query: str) -> str:
        """处理用户查询"""
        orchestrator = self.message_bus.agents["orchestrator"]
        return await orchestrator.handle_user_query(user_query)
```

## 🔄 **MCP + A2A 混合架构**

```python
class HybridSearchSystem:
    """结合MCP和A2A的混合系统"""
    
    def __init__(self):
        # MCP服务器提供工具
        self.mcp_server = SearchMCPServer()
        
        # A2A系统处理协作
        self.a2a_system = A2AWebSearchSystem()
        
        # 增强的Agent可以调用MCP工具
        self._enhance_agents_with_mcp()
    
    def _enhance_agents_with_mcp(self):
        """让A2A中的Agent能够调用MCP工具"""
        # 为每个Agent添加MCP客户端
        for agent in self.a2a_system.agents:
            agent.mcp_client = MCPClient()
            agent.mcp_client.connect("http://localhost:3000")
```

