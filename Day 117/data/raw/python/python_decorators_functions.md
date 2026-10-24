# Python Decorators, Closures, and First-Class Functions

In Python, functions are first-class citizens, meaning they can be passed as arguments, returned from other functions, assigned to variables, and stored in collections. A closure is a function that retains access to variables in its enclosing lexical scope even after that outer scope has finished execution.

A decorator is a callable that accepts another function as an argument, wraps or extends its behavior without modifying the underlying source code, and returns the modified wrapper function. Decorators commonly implement cross-cutting concerns such as logging, execution timing, input validation, authentication, and caching. Using functools.wraps preserves the decorated function's original docstring, name, and signature introspection.
