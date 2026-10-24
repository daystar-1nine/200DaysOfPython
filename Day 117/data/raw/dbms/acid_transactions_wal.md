# ACID Properties and Write-Ahead Logging in Relational Databases

Relational database management systems (RDBMS) guarantee transactional reliability through ACID properties. Atomicity ensures that all operations in a transaction execute successfully or the entire transaction aborts with zero side effects. Consistency guarantees that transactions transition the database from one valid state to another, preserving integrity constraints.

Isolation ensures concurrent transactions execute without interference, governed by isolation levels: Read Uncommitted, Read Committed, Repeatable Read, and Serializable. Durability guarantees that committed modifications persist permanently even across catastrophic system crashes. Durability is implemented via Write-Ahead Logging (WAL), where log records describing modifications are flushed to non-volatile disk before dirty memory pages are written to the database data files.
