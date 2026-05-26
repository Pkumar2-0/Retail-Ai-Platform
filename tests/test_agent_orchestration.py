from agents.analyst_agent import AGENT_NAME as ANALYST_NAME
from agents.azure_agent import AzureAgentResult
from agents.document_agent import AGENT_NAME as DOCUMENT_NAME
from agents.ml_agent import AGENT_NAME as ML_NAME
from agents.orchestrator import select_agents, synthesize_response
from agents.schemas import AgentMessage


def test_select_agents_for_cross_function_retail_question():
    selected = select_agents(
        "Compare total sales, inventory policy documents, and anomaly model usage"
    )

    assert selected == [ML_NAME, DOCUMENT_NAME, ANALYST_NAME]


def test_synthesize_response_combines_multiple_agent_messages():
    response = synthesize_response(
        "How should we plan inventory?",
        [
            AgentMessage(
                sender=ANALYST_NAME,
                receiver="Orchestrator Agent",
                intent="analytics_data_answer",
                content="Sales are highest in December.",
            ),
            AgentMessage(
                sender=DOCUMENT_NAME,
                receiver="Orchestrator Agent",
                intent="rag_document_answer",
                content="Restock based on forecast demand.",
            ),
        ],
    )

    assert "Multi-agent response" in response
    assert "Sales are highest in December." in response
    assert "Restock based on forecast demand." in response


def test_synthesize_response_prefers_azure_result_when_available():
    response = synthesize_response(
        "What should we do next?",
        [
            AgentMessage(
                sender=ANALYST_NAME,
                receiver="Orchestrator Agent",
                intent="analytics_data_answer",
                content="Local fallback content.",
            )
        ],
        azure_result=AzureAgentResult(
            enabled=True,
            response="Azure synthesized answer.",
        ),
    )

    assert response == "Azure synthesized answer."
