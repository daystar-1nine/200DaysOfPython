# Deadlock Detection and Prevention in Relational Engines

A deadlock occurs when two or more transactions maintain exclusive locks on resources while waiting to acquire locks held by each other, forming a cyclic dependency graph.
Neither transaction can proceed without external intervention.

Relational database engines resolve deadlocks using wait-for-graph cycle detection algorithms.
When a cycle is detected, the database engine aborts one of the transactions (the deadlock victim) and raises an error, allowing the other transaction to proceed.
Developers can prevent deadlocks by always acquiring table locks in a deterministic, global alphabetical order and keeping transactions short.