# ==============================================================================
# Program    : Library Book System
# Objective  : Model library books with discount application features.
# Concept    : OOP Encapsulation & Attribute Mutation Methods
# Why Used   : Encapsulates title, author, price, and applies percentages to mutate prices.
# ==============================================================================

# What is used : Class definition 'class Book:'
class Book:
    """Class representing a library book with pricing features."""

    def __init__(self, title, author, price):
        # What is used : Instance attributes
        self.title = title
        self.author = author
        self.price = price
        self.original_price = price

    # What is used : Instance method apply_discount(percentage)
    # Why it is used: Calculates and mutates book price based on discount percentage
    # How it works : Discounted = price - (price * percentage / 100)
    def apply_discount(self, percentage):
        if percentage < 0 or percentage > 100:
            print("Discount Error: Percentage must be between 0% and 100%!")
            return
        discount_amount = (self.original_price * percentage) / 100
        self.price = self.original_price - discount_amount
        print(f"Applied {percentage}% discount! Price reduced by Rs.{discount_amount:.2f}")

    # What is used : Instance method display_details()
    def display_details(self):
        print("\n====================================")
        print("          LIBRARY BOOK DETAILS      ")
        print("====================================")
        print(f"Title            : {self.title}")
        print(f"Author           : {self.author}")
        print(f"Original Price   : Rs.{self.original_price:,.2f}")
        print(f"Current Price    : Rs.{self.price:,.2f}")
        print("====================================")

def main():
    # Instantiating Book object
    book = Book("Python Basics", "Guido van Rossum", 650.0)

    book.display_details()
    print("\n[Action] Applying 10% promotional discount...")
    book.apply_discount(10)
    book.display_details()

if __name__ == "__main__":
    main()
