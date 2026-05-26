from fastapi import APIRouter
from backend.models.product_model import Product
from backend.config.database import database

router = APIRouter()

# Add Product API
@router.post("/add-product")
async def add_product(product: Product):

    product_dict = product.dict()

    result = await database.products.insert_one(product_dict)

    return {
        "message": "Product added successfully",
        "id": str(result.inserted_id)
    }

# Get Products API
@router.get("/products")
async def get_products():

    products = []

    async for product in database.products.find():

        product["_id"] = str(product["_id"])

        products.append(product)

    return products