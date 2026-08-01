# ==============================================================================
# Program    : Online Shopping Cart Analyzer (Bonus Challenge)
# Objective  : Analyze e-commerce shopping cart using functional tools & comprehensions.
# Concept    : Functional E-Commerce Pipeline (map, filter, reduce)
# Why Used   : Computes bill totals, GST tax (18%), filters premium items, applies discounts.
# ==============================================================================

from functools import reduce

cart = [
    {"item": "Laptop", "price": 65000},
    {"item": "Mouse", "price": 800},
    {"item": "Keyboard", "price": 1200},
    {"item": "Monitor", "price": 15000},
    {"item": "USB Cable", "price": 450}
]

def main():
    print("==========================================================================")
    print("                    ONLINE SHOPPING CART ANALYZER                         ")
    print("==========================================================================")

    # 1. Filter items with price above Rs.1000
    # What is used : filter() with lambda
    items_above_1000 = list(filter(lambda item: item["price"] > 1000, cart))
    print("\n--- 1. Premium Items (Price > Rs.1,000) ---")
    for item in items_above_1000:
        print(f"Item: {item['item']:<15} | Price: Rs.{item['price']:,.2f}")

    # 2. Apply 10% discount on items priced above Rs.5000
    # What is used : map() applying discount conditionally
    discounted_cart = list(map(
        lambda item: {
            "item": item["item"],
            "price": round(item["price"] * 0.90, 2) if item["price"] > 5000 else item["price"]
        },
        cart
    ))
    print("\n--- 2. Cart Items After Conditional Discounts ---")
    for item in discounted_cart:
        print(f"Item: {item['item']:<15} | Final Price: Rs.{item['price']:,.2f}")

    # 3. Calculate Subtotal Bill using reduce()
    # What is used : reduce() accumulator
    subtotal = reduce(lambda acc, item: acc + item["price"], discounted_cart, 0.0)

    # 4. Calculate 18% GST Tax
    gst_tax = subtotal * 0.18
    grand_total = subtotal + gst_tax

    print("\n==========================================================================")
    print("                        FINAL CHECKOUT INVOICE                            ")
    print("==========================================================================")
    print(f"Subtotal (Post-Discount) : Rs.{subtotal:,.2f}")
    print(f"GST Tax (18%)            : Rs.{gst_tax:,.2f}")
    print(f"GRAND TOTAL PAYABLE      : Rs.{grand_total:,.2f}")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
