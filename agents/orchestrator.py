from agents.analyst_agent import AGENT_NAME as ANALYST_NAME
from agents.analyst_agent import run_analyst_agent
from agents.azure_agent import AGENT_NAME as AZURE_NAME
from agents.azure_agent import synthesize_with_azure
from agents.document_agent import AGENT_NAME as DOCUMENT_NAME
from agents.document_agent import run_document_agent
from agents.ml_agent import AGENT_NAME as ML_NAME
from agents.ml_agent import run_ml_agent
from agents.schemas import AgentMessage


ORCHESTRATOR_NAME = "Orchestrator Agent"


def select_agents(query):

    query_lower = query.lower()

    document_keywords = [
        "policy",
        "document",
        "guide",
        "manual",
        "report",
        "procedure",
        "compliance",
        "regulation",
        "audit",
    ]

    ml_keywords = [
        "forecast",
        "prediction",
        "anomaly",
        "machine learning",
        "model",
        "accuracy",
        "training",
        "evaluation",
    ]

    selected_agents = []

    if any(keyword in query_lower for keyword in ml_keywords):
        selected_agents.append(ML_NAME)

    if any(keyword in query_lower for keyword in document_keywords):
        selected_agents.append(DOCUMENT_NAME)

    analytics_keywords = [
        "sales",
        "store",
        "holiday",
        "revenue",
        "average",
        "total",
        "dataset",
        "trend",
        "month",
        "performance",
    ]

    if any(keyword in query_lower for keyword in analytics_keywords):
        selected_agents.append(ANALYST_NAME)

    if not selected_agents:
        selected_agents.append(ANALYST_NAME)

    return selected_agents


def synthesize_response(query, agent_messages, azure_result=None):
    if azure_result and azure_result.response:
        return azure_result.response

    if len(agent_messages) == 1:
        return agent_messages[0].content

    sections = [
        f"{message.sender}: {message.content}"
        for message in agent_messages
    ]

    return (
        "Multi-agent response:\n"
        f"Question: {query}\n\n"
        + "\n\n".join(sections)
        + "\n\nCoordinator summary: Use the analytics answer for numeric sales context, "
        "the document answer for policy/process grounding, and the ML answer for model-driven interpretation."
    )


def orchestrator(query):
    selected_agents = select_agents(query)
    selected_agents_with_coordinator = [AZURE_NAME] + selected_agents

    messages = [
        AgentMessage(
            sender=ORCHESTRATOR_NAME,
            receiver=agent_name,
            intent="delegate_task",
            content=query,
            metadata={"reason": "Selected by keyword intent routing"},
        )
        for agent_name in selected_agents
    ]

    agent_runners = {
        ANALYST_NAME: run_analyst_agent,
        DOCUMENT_NAME: run_document_agent,
        ML_NAME: run_ml_agent,
    }
    agent_messages = [
        agent_runners[agent_name](query)
        for agent_name in selected_agents
    ]

    messages.extend(agent_messages)

    azure_result = synthesize_with_azure(query, agent_messages)
    if azure_result.enabled:
        messages.append(
            AgentMessage(
                sender=ORCHESTRATOR_NAME,
                receiver=AZURE_NAME,
                intent="synthesize_specialist_outputs",
                content=query,
                metadata={"input_agents": selected_agents},
            )
        )
        messages.append(
            AgentMessage(
                sender=AZURE_NAME,
                receiver=ORCHESTRATOR_NAME,
                intent="final_synthesis",
                content=azure_result.response or "",
                metadata={
                    "provider": "Azure OpenAI",
                    "status": "success" if azure_result.response else "fallback",
                    "error": azure_result.error,
                },
            )
        )

    response = synthesize_response(query, agent_messages, azure_result=azure_result)

    return {
        "agent": ORCHESTRATOR_NAME,
        "selected_agents": selected_agents_with_coordinator,
        "response": response,
        "messages": [message.to_dict() for message in messages],
        "features": {
            "prompt_engineering": True,
            "embeddings": True,
            "vector_store": "Azure AI Search if configured, otherwise ChromaDB",
            "rag": DOCUMENT_NAME in selected_agents,
            "multi_agent_orchestration": True,
            "mcp_ready_message_contract": True,
            "azure_genai_agent": azure_result.enabled,
            "azure_genai_status": "success" if azure_result.response else "fallback",
            "azure_genai_error": azure_result.error,
        },
    }
