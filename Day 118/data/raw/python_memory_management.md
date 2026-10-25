# Python Memory Management and Garbage Collection

Python manages memory using an internal private heap managed by the PyMalloc allocator for small allocations up to 512 bytes.
The primary memory reclamation mechanism is reference counting. Every Python object maintains an `ob_refcnt` header.
When `ob_refcnt` drops to zero, the memory occupied by the object is deallocated immediately.

To handle cyclic references that reference counting cannot resolve, Python incorporates a generational cyclical garbage collector (GC).
The GC organizes objects into three generations (Gen 0, Gen 1, Gen 2) based on object survival across collection cycles.
Gen 0 is inspected frequently, whereas Gen 2 contains long-lived objects.
A common source of memory leaks in Python is retaining references in global dictionaries, unclosed file descriptors, or cycles containing custom `__del__` implementations prior to Python 3.4.