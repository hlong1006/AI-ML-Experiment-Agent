"""
Configuration — đọc từ .env, fallback về defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # App
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR}/experiment_agent.db"
    )

    # Storage paths
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    MODELS_DIR: Path = BASE_DIR / os.getenv("MODELS_DIR", "saved_models")
    REPORTS_DIR: Path = BASE_DIR / os.getenv("REPORTS_DIR", "reports")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", str(BASE_DIR / "mlruns")
    )
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "ml-experiment-agent"
    )

    # Agent defaults
    MAX_EXPERIMENTS: int = int(os.getenv("MAX_EXPERIMENTS", "10"))
    TIME_BUDGET_MINUTES: int = int(os.getenv("TIME_BUDGET_MINUTES", "30"))
    DEFAULT_METRIC: str = os.getenv("DEFAULT_METRIC", "f1")

    def __init__(self):
        # Tạo thư mục cần thiết
        for d in [self.DATA_DIR / "raw", self.DATA_DIR / "processed",
                  self.MODELS_DIR, self.REPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
