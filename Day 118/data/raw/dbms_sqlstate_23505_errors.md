# Handling SQLSTATE 23505 Unique Constraint Violations

In relational databases (especially PostgreSQL and ANSI SQL compliant engines), error code SQLSTATE 23505 designates a unique constraint violation.
This error is triggered when an `INSERT` or `UPDATE` statement attempts to write a duplicate value into a column or composite columns constrained by a `UNIQUE` index or `PRIMARY KEY`.

To handle SQLSTATE 23505 gracefully without rolling back an entire application transaction, developers utilize `INSERT ... ON CONFLICT (column_name) DO UPDATE` or `DO NOTHING` in PostgreSQL (upsert pattern).
In distributed databases, mitigating unique constraint collisions often requires distributed ID generation schemes such as Snowflake IDs or ULIDs.