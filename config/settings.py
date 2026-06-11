"""Loads project settings from the .env file."""

import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Environment-driven application settings."""

    def __init__(self):
        
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "agentshield")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "agentshield_secret_2024")
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "agentshield_db")
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

        
        self.DATABASE_URL = (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

        
        self.ASYNC_DATABASE_URL = (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

        
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
        self.WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

        
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

        
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        
        self.RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
        self.RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
        self.RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
        self.CONVERSATION_MEMORY_SECONDS = int(
            os.getenv("CONVERSATION_MEMORY_SECONDS", "90")
        )

        
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.REDIS_ALLOW_FAKE = os.getenv("REDIS_ALLOW_FAKE", "0").lower() in {
            "1",
            "true",
            "yes",
        }

        
        self.APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
        self.APP_PORT = int(os.getenv("APP_PORT", "8080"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.CORS_ORIGINS = [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:8081,http://127.0.0.1:8081",
            ).split(",")
            if origin.strip()
        ]

    def print_config(self):
        """prints config settings to make sure it loaded right. doesn't print passwords tho"""
        print("=" * 50)
        print("  AgentShield - Configuration")
        print("=" * 50)
        print(f"  DB Host:        {self.POSTGRES_HOST}:{self.POSTGRES_PORT}")
        print(f"  DB Name:        {self.POSTGRES_DB}")
        print(f"  DB User:        {self.POSTGRES_USER}")
        print(f"  Whisper Model:  {self.WHISPER_MODEL}")
        print(f"  Whisper Device: {self.WHISPER_DEVICE}")
        print(f"  LLM Provider:   {self.LLM_PROVIDER}")
        print(f"  LLM Model:      {self.LLM_MODEL}")
        print(f"  Embedding:      {self.EMBEDDING_MODEL}")
        print(f"  RAG Chunk Size: {self.RAG_CHUNK_SIZE}")
        print(f"  RAG Top K:      {self.RAG_TOP_K}")
        print(f"  Memory Window:  {self.CONVERSATION_MEMORY_SECONDS}s")
        print(f"  Redis URL:      {self.REDIS_URL}")  
        print(f"  Server:         {self.APP_HOST}:{self.APP_PORT}")
        print(f"  Log Level:      {self.LOG_LEVEL}")
        
        print(f"  API Key Set:    {'Yes' if self.OPENAI_API_KEY else 'No'}")
        print("=" * 50)



if __name__ == "__main__":
    settings = Settings()
    settings.print_config()
