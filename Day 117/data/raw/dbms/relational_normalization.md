# Relational Database Normalization: 1NF, 2NF, 3NF, and BCNF

Normalization organizes database relational schemas to minimize data redundancy and prevent insertion, update, and deletion anomalies. First Normal Form (1NF) requires atomic attribute values and unique primary keys. Second Normal Form (2NF) builds upon 1NF by eliminating partial dependencies, ensuring that all non-key attributes depend on the complete composite primary key.

Third Normal Form (3NF) eliminates transitive dependencies, requiring that non-key attributes depend only on candidate keys (no non-key attribute determines another non-key attribute). Boyce-Codd Normal Form (BCNF) strictly requires that for every functional dependency X -> Y, X must be a superkey. Higher normal forms reduce anomalies at the expense of requiring additional SQL joins during retrieval.
