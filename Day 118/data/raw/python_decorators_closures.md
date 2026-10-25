# Python Decorators and Lexical Closures

In Python, functions are first-class citizens that can be passed as arguments, returned from other functions, and bound to variables.
A closure occurs when a nested inner function retains references to variables from its enclosing lexical scope even after the outer function has completed execution.

Decorators leverage closures to wrap functions or methods, modifying their behavior dynamically without altering the original source code.
Using `functools.wraps` preserves the wrapped function's metadata, including docstring, name, and parameter signatures.