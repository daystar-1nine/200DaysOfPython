# ==============================================================================
# Program    : Book Class Implementation
# Objective  : Model published books with titles, authors, and page counts.
# Concept    : Attributes & Formatted Instance Methods
# Why Used   : Encapsulates literary metadata into discrete book objects.
# ==============================================================================

# What is used : Class definition 'class Book:'
class Book:
    """Class representing a published book."""

    # What is used : Constructor __init__
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    # What is used : Instance method display_summary()
    def display_summary(self):
        print(f"[Book] Title: '{self.title}' | Author: {self.author} | Pages: {self.pages}")

def main():
    print("=== BOOK CATALOG ===")
    book1 = Book("Python Crash Course", "Eric Matthes", 544)
    book2 = Book("Clean Code", "Robert C. Martin", 464)

    book1.display_summary()
    book2.display_summary()

if __name__ == "__main__":
    main()
