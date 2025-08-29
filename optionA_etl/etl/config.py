from dataclasses import dataclass, field
from dotenv import load_dotenv; load_dotenv()
from .utils import env_str, env_int
@dataclass
class SECConfig:
    user_agent: str = env_str("SEC_USER_AGENT","Example Contact email@example.com")
    max_rps: int = env_int("SEC_REQUESTS_PER_SECOND",9)
@dataclass
class AppConfig:
    sec: SECConfig = field(default_factory=SECConfig)
