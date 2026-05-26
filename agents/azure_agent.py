import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


AGENT_NAME = "Azure GenAI Agent"


@dataclass
class AzureAgentResult:
    enabled: bool
    response: str | None = None
    error: str | None = None


def _load_azure_settings():
    load_dotenv()

    return {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    }


def is_azure_agent_configured():
    settings = _load_azure_settings()
    return all(settings.values())


def _build_client(endpoint, api_key):
    return OpenAI(
        api_key=api_key,
        base_url=endpoint.rstrip("/"),
    )


def synthesize_with_azure(query, agent_messages):
    settings = _load_azure_settings()
    if not all(settings.values()):
        return AzureAgentResult(
            enabled=False,
            error="Azure OpenAI settings are missing. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME.",
        )

    context = "\n\n".join(
        f"{message.sender} ({message.intent}):\n{message.content}"
        for message in agent_messages
    )

    system_prompt = (
        "You are the Azure GenAI coordinator for a retail AI platform. "
        "Synthesize specialist agent outputs into a concise, business-ready answer. "
        "Use retrieved documents as grounding when available, preserve numeric analytics exactly, "
        "and clearly separate recommendations from evidence."
    )

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Specialist agent outputs:\n{context}\n\n"
        "Return a final answer with the most relevant insight, supporting evidence, and suggested next action."
    )

    try:
        client = _build_client(settings["endpoint"], settings["api_key"])
        completion = client.chat.completions.create(
            model=settings["deployment"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return AzureAgentResult(
            enabled=True,
            response=completion.choices[0].message.content.strip(),
        )
    except Exception as exc:
        return AzureAgentResult(
            enabled=True,
            error=f"Azure GenAI Agent could not complete the request: {exc}",
        )
