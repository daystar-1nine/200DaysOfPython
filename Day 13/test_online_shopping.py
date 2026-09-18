import pytest
from online_shopping_system import Product, Electronics, Clothing, Grocery, ShoppingCart

def test_product_encapsulation():
    """Verify private price is protected and constrained to >= 0."""
    p = Product("1", "Test", 100)
    assert p.get_price() == 100
    
    p.set_price(-50) # Should fail silently based on logic
    assert p.get_price() == 100 # Remained 100
    
    p.set_price(200)
    assert p.get_price() == 200

def test_electronics_discount_polymorphism():
    """Verify electronics receives 10% discount."""
    e = Electronics("E1", "Laptop", 1000)
    assert e.calculate_discounted_price() == pytest.approx(900.0)

def test_clothing_discount_polymorphism():
    """Verify clothing receives 20% discount."""
    c = Clothing("C1", "Shirt", 1000, size="M")
    assert c.calculate_discounted_price() == pytest.approx(800.0)

def test_grocery_discount_polymorphism():
    """Verify grocery receives 5% discount."""
    g = Grocery("G1", "Apples", 1000, expiry_days=5)
    assert g.calculate_discounted_price() == pytest.approx(950.0)

def test_shopping_cart_checkout():
    """Verify total checkout calculates polymorphic sums correctly."""
    cart = ShoppingCart()
    cart.add_product(Electronics("E1", "Laptop", 1000))
    cart.add_product(Clothing("C1", "Shirt", 1000, size="M"))
    
    # We test state rather than stdout, but we can verify it doesn't crash
    # and cart holds correct logic. 
    assert len(cart.cart) == 2
    
    total_original = sum(item.get_price() for item in cart.cart)
    total_discounted = sum(item.calculate_discounted_price() for item in cart.cart)
    
    assert total_original == 2000.0
    assert total_discounted == pytest.approx(1700.0)
