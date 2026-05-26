from fastapi import FastAPI
from backend.config.database import database

from backend.routes.product_routes import router as product_router
from backend.routes.sales_routes import router as sales_router
from backend.routes.forecast_routes import router as forecast_router
from backend.routes.anomaly_routes import router as anomaly_router
from backend.routes.rag_routes import router as rag_router
from backend.routes.agent_routes import router as agent_router

app = FastAPI(
    title="Retail AI Platform",
    version="1.0"
)

@app.get("/")
async def home():

    collections = await database.list_collection_names()

    return {
        "message": "Retail AI Platform Running Successfully",
        "collections": collections
    }

app.include_router(product_router)
app.include_router(sales_router)
app.include_router(forecast_router)
app.include_router(anomaly_router)
app.include_router(rag_router)
app.include_router(agent_router)