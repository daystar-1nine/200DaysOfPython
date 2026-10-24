# Database Indexing: B-Trees, B+Trees, and Hash Indexes

Database indexes accelerate record retrieval queries at the cost of additional write latency and storage overhead. The B+Tree index is the standard indexing structure for relational engines. It maintains a balanced, multi-way search tree where internal nodes store guiding routing keys and leaf nodes store actual record pointers or table row data.

B+Trees store all data pointers exclusively in leaf nodes, which are linked together in a bidirectional doubly-linked list. This architecture makes B+Trees exceptionally efficient for range scans, prefix searches, and ordered sorting (ORDER BY) in O(log N) time. In contrast, Hash Indexes provide O(1) point lookups based on bucket hashing but cannot support range queries or sorting.
