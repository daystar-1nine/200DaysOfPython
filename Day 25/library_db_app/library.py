# ==============================================================================
# Program    : CLI Library Management System (Bonus Challenge)
# Objective  : Build multi-table database library system enforcing borrowing business rules.
# Concept    : Relational Database Schemas, Foreign Keys & Business Validation
# Why Used   : Connects books, members, and borrowings tables to prevent double borrowing of books.
# ==============================================================================

import argparse
from datetime import datetime
import logging
import os
import sqlite3
import sys

DB_FILE = os.path.join(os.path.dirname(__file__), "library.db")
LOG_FILE = os.path.join(os.path.dirname(__file__), "library_db.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def init_db() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Books Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                is_borrowed INTEGER DEFAULT 0
            )
        """)
        # Members Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)
        # Borrowings Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS borrowings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                member_id INTEGER NOT NULL,
                borrowed_date TEXT NOT NULL,
                returned_date TEXT,
                FOREIGN KEY (book_id) REFERENCES books (id),
                FOREIGN KEY (member_id) REFERENCES members (id)
            )
        """)
        conn.commit()

def handle_add_book(title: str, author: str) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO books (title, author) VALUES (?, ?)", (title, author))
        conn.commit()
        book_id = cursor.lastrowid
        logging.info("Added Book ID %d: '%s' by %s", book_id, title, author)
        print(f"[SUCCESS] Added Book #{book_id}: '{title}' by {author}")

def handle_list_books() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, author, is_borrowed FROM books ORDER BY id ASC")
        records = cursor.fetchall()
        if not records:
            print("No books found in library.")
            return
        print("\n------------------ LIBRARY BOOKS ------------------")
        print(f"{'ID':<4} | {'Title':<25} | {'Author':<20} | {'Status':<10}")
        print("-" * 65)
        for r in records:
            status = "BORROWED" if r[3] == 1 else "AVAILABLE"
            print(f"{r[0]:<4} | {r[1]:<25} | {r[2]:<20} | {status:<10}")
        print("---------------------------------------------------\n")

def handle_add_member(name: str) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO members (name) VALUES (?)", (name,))
        conn.commit()
        member_id = cursor.lastrowid
        logging.info("Added Member ID %d: %s", member_id, name)
        print(f"[SUCCESS] Added Member #{member_id}: {name}")

def handle_borrow(book_id: int, member_id: int) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        
        # Check if book exists and is available
        cursor.execute("SELECT id, title, is_borrowed FROM books WHERE id = ?", (book_id,))
        book = cursor.fetchone()
        if not book:
            print(f"[ERROR] Book ID #{book_id} does not exist.")
            return
        if book[2] == 1:
            logging.warning("Borrow failed: Book ID %d is already borrowed", book_id)
            print(f"[BUSINESS LOGIC ERROR] Book #{book_id} ('{book[1]}') is ALREADY BORROWED!")
            return

        # Check if member exists
        cursor.execute("SELECT id, name FROM members WHERE id = ?", (member_id,))
        member = cursor.fetchone()
        if not member:
            print(f"[ERROR] Member ID #{member_id} does not exist.")
            return

        # Borrow execution
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO borrowings (book_id, member_id, borrowed_date) VALUES (?, ?, ?)", (book_id, member_id, now_str))
        cursor.execute("UPDATE books SET is_borrowed = 1 WHERE id = ?", (book_id,))
        conn.commit()

        logging.info("Book ID %d ('%s') borrowed by Member ID %d (%s)", book_id, book[1], member_id, member[1])
        print(f"[SUCCESS] Book #{book_id} ('{book[1]}') successfully borrowed by {member[1]}!")

def handle_return(book_id: int) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, is_borrowed FROM books WHERE id = ?", (book_id,))
        book = cursor.fetchone()
        if not book:
            print(f"[ERROR] Book ID #{book_id} does not exist.")
            return
        if book[2] == 0:
            print(f"[WARNING] Book #{book_id} ('{book[1]}') was not borrowed.")
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE borrowings SET returned_date = ? WHERE book_id = ? AND returned_date IS NULL", (now_str, book_id))
        cursor.execute("UPDATE books SET is_borrowed = 0 WHERE id = ?", (book_id,))
        conn.commit()

        logging.info("Book ID %d ('%s') returned to library", book_id, book[1])
        print(f"[SUCCESS] Book #{book_id} ('{book[1]}') successfully returned to library!")

def handle_search(query: str) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        search_pattern = f"%{query}%"
        cursor.execute("SELECT id, title, author, is_borrowed FROM books WHERE title LIKE ? OR author LIKE ?", (search_pattern, search_pattern))
        records = cursor.fetchall()
        print(f"\n--- SEARCH RESULTS FOR '{query}' ---")
        if not records:
            print("No matching books found.")
        else:
            for r in records:
                status = "BORROWED" if r[3] == 1 else "AVAILABLE"
                print(f"ID: {r[0]:<3} | Title: {r[1]:<25} | Author: {r[2]:<20} | Status: {status}")
        print("------------------------------------\n")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Library Management System")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # add-book
    ab_p = subparsers.add_parser("add-book", help="Add new book")
    ab_p.add_argument("--title", type=str, required=True, help="Book title")
    ab_p.add_argument("--author", type=str, required=True, help="Book author")

    # list-books
    lb_p = subparsers.add_parser("list-books", help="List all books")

    # add-member
    am_p = subparsers.add_parser("add-member", help="Add new member")
    am_p.add_argument("--name", type=str, required=True, help="Member name")

    # borrow
    bw_p = subparsers.add_parser("borrow", help="Borrow a book")
    bw_p.add_argument("--book-id", type=int, required=True, help="Book ID")
    bw_p.add_argument("--member-id", type=int, required=True, help="Member ID")

    # return-book
    rb_p = subparsers.add_parser("return-book", help="Return a borrowed book")
    rb_p.add_argument("--book-id", type=int, required=True, help="Book ID")

    # search
    sr_p = subparsers.add_parser("search", help="Search books by title/author")
    sr_p.add_argument("--query", type=str, required=True, help="Search query")

    return parser

def main() -> None:
    print("=== BONUS CHALLENGE: LIBRARY MANAGEMENT SYSTEM ===")
    init_db()
    parser = create_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        handle_add_book("Python Crash Course", "Eric Matthes")
        handle_add_book("Clean Code", "Robert Martin")
        handle_add_member("Suraj Sawant")
        handle_list_books()
        handle_borrow(1, 1)
        handle_borrow(1, 1)  # Trigger business logic prevention error
        handle_return(1)
        handle_search("Python")
    else:
        args = parser.parse_args()
        if args.command == "add-book":
            handle_add_book(args.title, args.author)
        elif args.command == "list-books":
            handle_list_books()
        elif args.command == "add-member":
            handle_add_member(args.name)
        elif args.command == "borrow":
            handle_borrow(args.book_id, args.member_id)
        elif args.command == "return-book":
            handle_return(args.book_id)
        elif args.command == "search":
            handle_search(args.query)

if __name__ == "__main__":
    main()
