"""Small, platform-neutral settings for one agent execution."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path


_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "agent.md"
INSTRUCTIONS = _PROMPT.read_text(encoding="utf-8") if _PROMPT.exists() else ""
PROMPT_HASH = sha256(INSTRUCTIONS.encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    live_api_enabled: bool = False
    interrupts_enabled: bool = True

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        enabled = environment.get("SIMPLE_AGENT_ENABLE_LIVE_API", "0")
        if enabled not in {"0", "1"}:
            raise ValueError("SIMPLE_AGENT_ENABLE_LIVE_API must be 0 or 1")
        return cls(live_api_enabled=enabled == "1")

    def public(self):
        return {**asdict(self), "promptHash": PROMPT_HASH}
