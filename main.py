import asyncio
from mcp_search_server import SearchMCPServer
from a2a_agents import MessageBus, SearchAgent, ContentAgent, AnalysisAgent, SynthesisAgent, OrchestratorAgent

async def run_mcp_server():
    server = SearchMCPServer()
    server.run()

def run_mcp():
    import threading
    t = threading.Thread(target=run_mcp_server, daemon=True)
    t.start()

class A2AWebSearchSystem:
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents = []
        self._setup_agents()
    def _setup_agents(self):
        search_agent = SearchAgent("search_agent", self.message_bus)
        content_agent = ContentAgent("content_agent", self.message_bus)
        analysis_agent = AnalysisAgent("analysis_agent", self.message_bus)
        synthesis_agent = SynthesisAgent("synthesis_agent", self.message_bus)
        orchestrator = OrchestratorAgent("orchestrator", self.message_bus)
        for agent in [search_agent, content_agent, analysis_agent, synthesis_agent, orchestrator]:
            self.message_bus.register_agent(agent)
            self.agents.append(agent)
    async def start(self):
        tasks = [agent.run() for agent in self.agents]
        await asyncio.gather(*tasks)
    async def query(self, user_query: str) -> str:
        orchestrator = self.message_bus.agents["orchestrator"]
        return await orchestrator.handle_user_query(user_query)

async def main():
    run_mcp()
    a2a = A2AWebSearchSystem()
    asyncio.create_task(a2a.start())
    await asyncio.sleep(1)
    answer = await a2a.query("什么是MCP协议？")
    print("最终答案：", answer)

if __name__ == "__main__":
    asyncio.run(main())
