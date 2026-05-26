from pydantic import BaseModel

class Sale(BaseModel):

    product_id: int
    date: str
    quantity: int
    revenue: float