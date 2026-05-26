from pydantic import BaseModel

class AnomalyInput(BaseModel):

    Weekly_Sales: float
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float