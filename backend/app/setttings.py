
import os

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
    RESEARCH_SERVER_CMD = [os.getenv("PYTHON", "python"), "-m", "mcp_servers.research.server"]

settings = Settings()