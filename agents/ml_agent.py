from agents.data_utils import get_ml_model_context
from agents.schemas import AgentMessage



AGENT_NAME = "ML Expert Agent"


def ml_agent(query):

    model_context = get_ml_model_context()
    return (
        "Retail machine learning context for Azure answer generation:\n"
        f"{model_context}\n\n"
        f"User question: {query}"
    )


def run_ml_agent(query, receiver="Orchestrator Agent"):
    return AgentMessage(
        sender=AGENT_NAME,
        receiver=receiver,
        intent="ml_model_insight",
        content=ml_agent(query),
        metadata={
            "source": "ml/saved_models",
            "capabilities": ["forecast interpretation", "anomaly detection guidance"],
        },
    )
