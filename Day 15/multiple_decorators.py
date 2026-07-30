# ==============================================================================
# Program    : Stacked Multiple Decorators
# Objective  : Demonstrate execution order when stacking multiple decorators.
# Concept    : Decorator Stacking (Bottom-Up Evaluation, Top-Down Wrapper Execution)
# Why Used   : Teaches how @bold and @italic apply in sequence around a string return value.
# ==============================================================================

import functools

# What is used : Bold HTML Decorator 'def bold(func)'
def bold(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

# What is used : Italic HTML Decorator 'def italic(func)'
def italic(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

# What is used : Stacked Decorators (@bold on top of @italic)
# How it works : Equivalent to get_message = bold(italic(get_message))
@bold
@italic
def get_message():
    return "Hello Stacked Decorators!"

def main():
    print("=== Multiple Decorators Demonstration ===")
    formatted_html = get_message()
    print("Formatted HTML Output:", formatted_html)

if __name__ == "__main__":
    main()
