# ACID Properties and Transaction Isolation Levels

Database transactions adhere to ACID properties: Atomicity (all-or-nothing execution), Consistency (maintaining valid schema invariants), Isolation (concurrent transactions execute without interference), and Durability (committed changes survive system crashes).

ANSI SQL defines four isolation levels: Read Uncommitted, Read Committed, Repeatable Read, and Serializable.
Higher isolation levels prevent concurrency anomalies such as Dirty Reads, Non-Repeatable Reads, and Phantom Reads, but may increase lock contention and serialization failures.
Modern relational databases like PostgreSQL utilize Multi-Version Concurrency Control (MVCC) to ensure readers do not block writers and writers do not block readers.