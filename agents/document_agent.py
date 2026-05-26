from agents.schemas import AgentMessage



AGENT_NAME = "Document Assistant Agent"


def _search_documents(query):
    from backend.services.rag_service import search_documents

    return search_documents(query)


def document_agent(query, rag_result=None):

    if rag_result is None:
        rag_result = _search_documents(query)
    documents = rag_result.get("documents", [])

    if not documents:
        return "I could not find any relevant retail documents for that query."

    snippets = [f"{idx + 1}. {doc}" for idx, doc in enumerate(documents[:3])]
    snippet_text = "\n\n".join(snippets)

    return (
        "Document search results:\n"
        f"Found {len(documents)} relevant document excerpt(s).\n\n"
        "Top retrieved snippets:\n"
        f"{snippet_text}\n\n"
        "Use these passages as the basis for retail policy or process guidance."
    )


def run_document_agent(query, receiver="Orchestrator Agent"):
    rag_result = _search_documents(query)
    documents = rag_result.get("documents", [])
    content = document_agent(query, rag_result=rag_result)

    return AgentMessage(
        sender=AGENT_NAME,
        receiver=receiver,
        intent="rag_document_answer",
        content=content,
        metadata={
            "source": rag_result.get("backend", "ChromaDB"),
            "fallback_reason": rag_result.get("fallback_reason"),
            "retrieved_chunks": len(documents),
            "documents": documents[:3],
            "capabilities": ["embeddings", "vector search", "RAG"],
        },
    )
