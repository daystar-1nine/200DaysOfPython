# ==============================================================================
# Program    : Exercise 4 - Products APIRouter (ex4_products_router.py)
# Objective  : Create an independent APIRouter for /products resource.
# Concept    : Modular APIRouter Component
# Why Used   : Demonstrates building independent resource routers cleanly.
# ==============================================================================

from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient
from pydantic import BaseModel

router = APIRouter(prefix="/products", tags=["Products"])

class Product(BaseModel):
    id: int
    title: str
    price: float

@router.get("", response_model=list[Product])
def list_products():
    return [
        Product(id=101, title="Mechanical Keyboard", price=79.99),
        Product(id=102, title="Wireless Mouse", price=29.99)
    ]

app = FastAPI(title="Exercise 4 - Products Router")
app.include_router(router)

if __name__ == "__main__":
    client = TestClient(app)
    res = client.get("/products")
    print(f"Exercise 4 Response: {res.json()}")
