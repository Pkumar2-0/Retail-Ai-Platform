# Azure Deployment Diagram

```mermaid
flowchart TD
    User["User / Swagger UI / Frontend"] --> App["FastAPI Backend on Azure Web App"]

    App --> Orchestrator["Orchestrator Agent"]
    Orchestrator --> Analyst["Data Analyst Agent"]
    Orchestrator --> Document["Document Assistant Agent"]
    Orchestrator --> ML["ML Expert Agent"]
    Orchestrator --> AzureAgent["Azure GenAI Agent"]

    Analyst --> Mongo["MongoDB Atlas Retail Data"]
    Document --> Search["Azure AI Search Vector Index"]
    Document --> Chroma["Local ChromaDB Fallback"]
    ML --> ModelFiles["Saved ML Models / Azure ML Optional"]
    AzureAgent --> OpenAI["Azure OpenAI Deployment: gpt-4.1-mini"]

    App --> Env["App Service Environment Variables"]
    Env --> Vault["Azure Key Vault Recommended"]
```

Azure components used:

- Azure OpenAI through Microsoft Foundry
- Azure AI Foundry for model deployment management
- Azure Web App / App Service for FastAPI deployment
- Azure AI Search for cloud RAG vector retrieval
