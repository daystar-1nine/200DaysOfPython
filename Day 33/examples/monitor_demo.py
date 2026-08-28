# ==============================================================================
# Program    : Function Monitoring System Interactive Demo
# Objective  : Demonstrate @logger, @timer, @retry, and @requires_auth stacked decorators.
# Concept    : Decorator Composition & Monitoring
# Why Used   : Validates decorator execution order and behavior in a real script.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from decorators.logger import logger
from decorators.timer import timer
from decorators.retry import retry
from decorators.auth import requires_auth

# Global counter for retry demo
attempt_counter = 0

@timer
@logger
def process_data(items: list[int]) -> int:
    """Process numbers and return sum."""
    return sum(items)

@retry(max_attempts=3, delay=0.01)
def unstable_api_call() -> str:
    global attempt_counter
    attempt_counter += 1
    if attempt_counter < 3:
        raise ConnectionError("Network glitch")
    return "API Success Result"

@requires_auth(role="admin")
def delete_user_database(user_id: int) -> str:
    return f"User #{user_id} deleted"

def main():
    print("==================================================")
    print("      DAY 33 - FUNCTION MONITORING SYSTEM         ")
    print("==================================================\n")

    print("--- 1. Testing Stacked @timer and @logger ---")
    result = process_data([10, 20, 30, 40])
    print(f"Result: {result}\n")

    print("--- 2. Testing Parameterized @retry ---")
    api_res = unstable_api_call()
    print(f"API Result: {api_res}\n")

    print("--- 3. Testing Authorization @requires_auth ---")
    auth_res = delete_user_database(42)
    print(f"Auth Result: {auth_res}\n")

if __name__ == "__main__":
    main()
