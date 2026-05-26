from agents.data_utils import get_ml_model_context, get_direct_ml_answer
from agents.schemas import AgentMessage



AGENT_NAME = "ML Expert Agent"


def ml_agent(query):

    direct_answer = get_direct_ml_answer(query)
    if direct_answer:
        return direct_answer

    model_context = get_ml_model_context()
    return (
        "Retail machine learning summary:\n"
        f"{model_context}\n\n"
        f"Your query: {query}\n"
        "This platform uses forecasting and anomaly detection to support retail planning and to surface unusual sales events."
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
