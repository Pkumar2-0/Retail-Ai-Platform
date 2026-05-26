from fastapi import APIRouter
from backend.models.sales_model import Sale
from backend.config.database import database

router = APIRouter()

# Add Sale API
@router.post("/add-sale")
async def add_sale(sale: Sale):

    sale_dict = sale.dict()

    result = await database.sales.insert_one(sale_dict)

    return {
        "message": "Sale added successfully",
        "id": str(result.inserted_id)
    }

# Get Sales API
@router.get("/sales")
async def get_sales():

    sales = []

    async for sale in database.sales.find():

        sale["_id"] = str(sale["_id"])

        sales.append(sale)

    return sales