from agents.data_utils import get_analytics_context
from agents.schemas import AgentMessage



AGENT_NAME = "Data Analyst Agent"


def analyst_agent(query):

    analytics_context = get_analytics_context()

    return (
        "Retail analytics context for Azure answer generation:\n"
        f"{analytics_context}\n\n"
        f"User question: {query}"
    )


def run_analyst_agent(query, receiver="Orchestrator Agent"):
    return AgentMessage(
        sender=AGENT_NAME,
        receiver=receiver,
        intent="analytics_data_answer",
        content=analyst_agent(query),
        metadata={
            "source": "ml/datasets",
            "capabilities": ["pandas analytics", "sales KPI summary"],
        },
    )
