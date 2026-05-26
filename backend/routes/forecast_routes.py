from fastapi import APIRouter

from backend.models.forecast_model import ForecastInput
from backend.services.forecast_service import predict_sales

router = APIRouter()

@router.post("/predict-sales")
async def forecast_sales(input_data: ForecastInput):

    prediction = predict_sales(
        input_data.dict()
    )

    return {
        "predicted_weekly_sales": prediction
    }