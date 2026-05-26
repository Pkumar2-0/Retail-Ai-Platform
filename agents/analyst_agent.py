from agents.data_utils import get_analytics_context, get_direct_analytics_answer
from agents.schemas import AgentMessage



AGENT_NAME = "Data Analyst Agent"


def analyst_agent(query):

    direct_answer = get_direct_analytics_answer(query)
    if direct_answer:
        return direct_answer

    analytics_context = get_analytics_context()

    return (
        "Retail analytics summary:\n"
        f"{analytics_context}\n\n"
        f"Your query: {query}\n"
        "Based on the available dataset, the retail sales story is driven by store-level demand, holiday lift, and economic factors."
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
