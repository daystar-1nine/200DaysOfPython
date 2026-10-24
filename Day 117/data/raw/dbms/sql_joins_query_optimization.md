# SQL Join Algorithms and Cost-Based Query Optimization

Relational database engines combine records across tables using join operations. Inner Joins return matching rows from both relations; Left Outer Joins return all rows from the left relation with matching rows from the right relation (filling unmatched attributes with NULLs).

The Cost-Based Query Optimizer evaluates alternative physical execution plans using catalog statistics (histograms, table row counts, index selectivity). It chooses between three fundamental join algorithms: Nested Loop Join (optimal when joining small subsets against an indexed table), Hash Join (optimal for large unindexed equijoins by building an in-memory hash table on the smaller relation), and Sort-Merge Join (optimal when both input relations are already sorted on the join key).
