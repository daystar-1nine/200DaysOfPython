# Database Indexing: B-Tree vs Hash Indexes

B-Tree indexes are balanced search trees where leaf nodes are sorted and linked sequentially.
They support equality lookups `(=)` as well as range scans `(<, <=, >, >=, BETWEEN)` in O(log N) time.
B-Trees are the default index structure in PostgreSQL, MySQL, and SQLite.

Hash indexes build an in-memory hash table of keys pointing to row IDs.
They offer O(1) average lookup time for exact equality searches but cannot accelerate range queries, sorting (`ORDER BY`), or partial prefix searches.