from pydantic import BaseModel

class ForecastInput(BaseModel):

    Store: int
    Holiday_Flag: int
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    Year: int
    Month: int
    Day: int
    Week: int