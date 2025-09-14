from mcp_search_server.py import SearchMCPServer
from a2a_agents import A2AWebSearchSystem

class MCPClient:
    def connect(self, url):
        pass

class HybridSearchSystem:
    def __init__(self):
        self.mcp_server = SearchMCPServer()
        self.a2a_system = A2AWebSearchSystem()
        self._enhance_agents_with_mcp()
    def _enhance_agents_with_mcp(self):
        for agent in self.a2a_system.agents:
            agent.mcp_client = MCPClient()
            agent.mcp_client.connect("http://localhost:3000")
