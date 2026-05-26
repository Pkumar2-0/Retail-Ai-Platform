from fastapi import APIRouter

from backend.models.anomaly_model import AnomalyInput
from backend.services.anomaly_service import detect_anomaly

router = APIRouter()

@router.post("/detect-anomaly")
async def anomaly_detection(
    input_data: AnomalyInput
):

    result = detect_anomaly(
        input_data.dict()
    )

    return {
        "result": result
    }