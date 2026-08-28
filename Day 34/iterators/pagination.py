# ==============================================================================
# Program    : Pagination Iterator (PaginationIterator)
# Objective  : Chunk dataset lists into distinct pages of size page_size.
# Concept    : Dataset Chunking Iterator
# Why Used   : Simulates web API and database query result pagination.
# ==============================================================================

class PaginationIterator:
    def __init__(self, data: list, page_size: int = 2):
        if page_size < 1:
            raise ValueError("Page size must be at least 1.")
        self.data = list(data)
        self.page_size = page_size
        self._index = 0
        self._page_number = 1

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._index >= len(self.data):
            raise StopIteration
        chunk = self.data[self._index : self._index + self.page_size]
        page_info = {
            "page": self._page_number,
            "items": chunk,
            "count": len(chunk)
        }
        self._index += self.page_size
        self._page_number += 1
        return page_info


if __name__ == "__main__":
    print("=== PAGINATION ITERATOR DEMO ===")
    sample_data = ["A", "B", "C", "D", "E"]
    for page in PaginationIterator(sample_data, page_size=2):
        print(f"Page {page['page']}: {page['items']} (Count: {page['count']})")
