from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    intent: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

