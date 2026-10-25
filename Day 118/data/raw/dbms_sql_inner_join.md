# SQL INNER JOIN Principles and Query Execution

The SQL INNER JOIN clause selects records that have matching values in both joined tables based on a specified join predicate.
The standard syntax is:
`SELECT columns FROM table_a INNER JOIN table_b ON table_a.foreign_key = table_b.primary_key;`

If a row in Table A does not match any row in Table B under the ON predicate, it is omitted from the resulting relation.
Relational query optimizers evaluate join strategies using algorithms such as Nested Loop Join, Hash Join, or Sort-Merge Join depending on table statistics and available indexes.
For large datasets with equi-join conditions, Hash Joins typically provide O(M + N) performance when an in-memory hash table fits in `work_mem`.